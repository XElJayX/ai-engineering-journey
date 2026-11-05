# Bridge lesson 7: Backpropagation. Build a tiny autograd engine, then train a network with it.
# Run: python3 labs/03-03-backpropagation/my_autograd.py   (stdlib only, no numpy)
# Rule for every node: input.grad += (local derivative) * out.grad
import math
import random


class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None   # set by the op that created this Value
        self._children = children       # the Values this one was computed from
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: the "swap" rule. d(a*b)/da = b, d(a*b)/db = a.
            # Careful: x * x passes the SAME Value as self and other. Your code must still give 2x.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, self.data))))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            # TODO: local derivative of sigmoid, reusing s from the forward pass.
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._children:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for v in reversed(topo):
            v._backward()
        


    # Provided helpers (built from your ops, so they work once yours do).
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


# ---------------------------------------------------------------------------
# Provided: a small network built on Value. Read it, don't edit it.
# ---------------------------------------------------------------------------

class Neuron:
    def __init__(self, n_in):
        scale = (2.0 / n_in) ** 0.5
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        z = self.b
        for wi, xi in zip(self.w, x):
            z = z + wi * xi
        return z.sigmoid()

    def parameters(self):
        return self.w + [self.b]


class MLP:
    def __init__(self, sizes):
        self.layers = [[Neuron(sizes[i]) for _ in range(sizes[i + 1])] for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


def train_xor(epochs=1000, lr=1.0, seed=42):
    random.seed(seed)
    net = MLP([2, 4, 1])
    data = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
    for _ in range(epochs):
        loss = Value(0.0)
        for x, y in data:
            diff = net([Value(v) for v in x]) - y
            loss = loss + diff * diff          # squared error, summed over the 4 examples
        net.zero_grad()
        loss.backward()
        for p in net.parameters():
            p.data -= lr * p.grad
    return net, loss.data, data


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def numeric_grad(f, xs, i, h=1e-6):
    # Nudge input i up and down and measure the slope: the slow way backprop replaces.
    up = [v + (h if j == i else 0.0) for j, v in enumerate(xs)]
    dn = [v - (h if j == i else 0.0) for j, v in enumerate(xs)]
    return (f(*up) - f(*dn)) / (2 * h)


def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_add_mul():
    x, y, z = Value(2.0), Value(-3.0), Value(4.0)
    f = x * y + z
    f.backward()
    got = (x.grad, y.grad, z.grad)
    return got == (-3.0, 2.0, 1.0), f"(x, y, z).grad = {got}, expected (-3.0, 2.0, 1.0)"


def t_reuse():
    x, y, z = Value(2.0), Value(-3.0), Value(4.0)
    f = x * y + x * z
    f.backward()
    return x.grad == 1.0, f"x.grad = {x.grad}, expected y + z = 1.0 (gradients from both paths must ADD)"


def t_square():
    x = Value(3.0)
    f = x * x
    f.backward()
    return x.grad == 6.0, f"d(x*x)/dx at 3 = {x.grad}, expected 6.0"


def t_sigmoid():
    x = Value(0.7)
    f = x.sigmoid()
    f.backward()
    num = numeric_grad(lambda a: 1 / (1 + math.exp(-a)), [0.7], 0)
    return abs(x.grad - num) < 1e-6, f"autograd {x.grad:.6f} vs numeric {num:.6f}"


def t_deep_expression():
    def build(a, b, c):
        a, b, c = Value(a), Value(b), Value(c)
        q = a * b + c
        r = (q * a).sigmoid()
        f = r * q + b * r - c
        return f, (a, b, c)

    xs = [0.5, -1.2, 0.8]
    f, vals = build(*xs)
    f.backward()
    worst = max(abs(v.grad - numeric_grad(lambda *t: build(*t)[0].data, xs, i)) for i, v in enumerate(vals))
    return worst < 1e-5, f"max |autograd - numeric| = {worst:.2e} (needs correct topological order)"


def t_network_grads():
    random.seed(0)
    net = MLP([3, 4, 1])
    x = [0.2, -0.5, 0.9]
    out = net([Value(v) for v in x])
    loss = (out - 1.0) * (out - 1.0)
    net.zero_grad()
    loss.backward()
    worst = 0.0
    for p in net.parameters():
        old = p.data
        p.data = old + 1e-6
        o = net([Value(v) for v in x]).data
        up = (o - 1.0) ** 2
        p.data = old - 1e-6
        o = net([Value(v) for v in x]).data
        dn = (o - 1.0) ** 2
        p.data = old
        worst = max(worst, abs(p.grad - (up - dn) / 2e-6))
    return worst < 1e-5, f"all {len(net.parameters())} weights: max |autograd - numeric| = {worst:.2e}"


def t_xor():
    net, loss, data = train_xor()
    preds = [net([Value(v) for v in x]).data for x, _ in data]
    right = all((p > 0.5) == (y == 1.0) for p, (_, y) in zip(preds, data))
    return loss < 0.05 and right, f"final loss {loss:.4f}, predictions {[round(p, 2) for p in preds]}"


def t_vanishing():
    random.seed(1)
    ws = [Value(random.uniform(0.5, 1.5)) for _ in range(10)]
    h = Value(0.5)
    for w in ws:
        h = (w * h).sigmoid()
    h.backward()
    first, last = abs(ws[0].grad), abs(ws[-1].grad)
    return first < last / 100, f"10 sigmoid layers: |grad| first layer {first:.2e} vs last layer {last:.2e}"


if __name__ == "__main__":
    tests = [
        ("+ and * on x*y + z", t_add_mul),
        ("x used twice: gradients accumulate", t_reuse),
        ("x * x: same Value as both inputs", t_square),
        ("sigmoid matches the numeric slope", t_sigmoid),
        ("deeper expression matches numeric gradients", t_deep_expression),
        ("every weight in a small MLP matches numeric", t_network_grads),
        ("trains XOR with your engine", t_xor),
        ("vanishing gradient: layer 1 gets < 1/100 of layer 10's gradient", t_vanishing),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
