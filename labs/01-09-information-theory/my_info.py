# Bridge lesson 4: Information Theory. Build entropy, cross-entropy, KL and perplexity.
# Run: python3 labs/01-09-information-theory/my_info.py   (stdlib only)
# All functions use NATURAL log (nats) unless a base is given, like PyTorch.
import math


def entropy(p, base=math.e):
    # H(P) = -sum p(x) log p(x). Terms with p(x) == 0 contribute 0 (they never happen).
    ent = 0
    for i in p:
        if i != 0:
            ent += (i * math.log(i, base))
    return - ent


def cross_entropy(p, q, base=math.e):
    # H(P, Q) = -sum p(x) log q(x).
    # Skip terms where p(x) == 0. If p(x) > 0 but q(x) == 0, return math.inf
    # (the model said "impossible" about something that happened).
    cross = 0
    for i in range(len(p)):
        if p[i] != 0:
            if q[i] == 0:
                return math.inf
            else:
                cross+= p[i] * math.log(q[i], base)
    return -cross

def kl_divergence(p, q, base=math.e):
    # D_KL(P || Q). One line, using the two functions above.
    return cross_entropy(p,q) - entropy(p)


def sequence_perplexity(token_logprobs):
    # token_logprobs: the model's natural-log probability of each ACTUAL next token in a sequence,
    # e.g. what an LLM API returns as `logprobs` for the tokens it generated or scored.
    # Return the perplexity of the whole sequence: exp(average negative log-prob).
    avg = 0
    for prob in token_logprobs:
        avg+=prob
    return math.exp(-avg/len(token_logprobs))
    


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_entropy_bits():
    fair, biased = entropy([0.5, 0.5], base=2), entropy([0.99, 0.01], base=2)
    return abs(fair - 1.0) < 1e-9 and abs(biased - 0.0808) < 1e-3, f"fair={fair:.4f} bits, biased={biased:.4f} bits"


def t_entropy_zero():
    h = entropy([1.0, 0.0, 0.0])
    return h == 0.0, f"one-hot entropy = {h}"


def t_entropy_uniform():
    h = entropy([1 / 50_000] * 50_000)
    return abs(h - math.log(50_000)) < 1e-6, f"uniform over 50k = {h:.4f} nats (ln 50000 = {math.log(50_000):.4f})"


def t_ce_onehot():
    ce = cross_entropy([1.0, 0.0, 0.0], [0.659, 0.242, 0.099])
    return abs(ce - 0.4170) < 1e-3, f"{ce:.4f} nats (matches last lesson's 0.4170)"


def t_ce_ge_entropy():
    p = [0.7, 0.2, 0.1]
    ce_good, ce_bad, h = cross_entropy(p, [0.6, 0.25, 0.15]), cross_entropy(p, [0.1, 0.1, 0.8]), entropy(p)
    return h < ce_good < ce_bad, f"H={h:.4f}, CE(good)={ce_good:.4f}, CE(bad)={ce_bad:.4f}"


def t_ce_inf():
    return cross_entropy([0.5, 0.5], [1.0, 0.0]) == math.inf, ""


def t_kl():
    p, q = [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]
    same, pq, qp = kl_divergence(p, p), kl_divergence(p, q), kl_divergence(q, p)
    return abs(same) < 1e-12 and pq > 0 and qp > 0 and abs(pq - qp) > 0.1, f"KL(P||P)={same:.1e}, KL(P||Q)={pq:.4f}, KL(Q||P)={qp:.4f}"


def t_perplexity():
    uniform8 = sequence_perplexity([math.log(1 / 8)] * 20)
    perfect = sequence_perplexity([0.0] * 5)
    mixed = sequence_perplexity([math.log(0.5), math.log(0.25), math.log(0.125)])
    return abs(uniform8 - 8) < 1e-9 and abs(perfect - 1) < 1e-12 and abs(mixed - 4) < 1e-9, \
        f"uniform-8 -> {uniform8:.4f}, perfect -> {perfect:.4f}, [1/2, 1/4, 1/8] -> {mixed:.4f}"


if __name__ == "__main__":
    tests = [
        ("entropy: fair coin = 1 bit, 99/1 coin ~ 0.08 bits", t_entropy_bits),
        ("entropy: one-hot (certain) = 0, no log(0) crash", t_entropy_zero),
        ("entropy: uniform over 50k tokens = ln(50000)", t_entropy_uniform),
        ("cross_entropy with one-hot P = -log q(true)", t_ce_onehot),
        ("cross_entropy >= entropy, worse model -> larger", t_ce_ge_entropy),
        ("cross_entropy = inf when q says impossible", t_ce_inf),
        ("KL: zero for P||P, positive, NOT symmetric", t_kl),
        ("perplexity: uniform-8 -> 8, perfect -> 1, mixed -> 4", t_perplexity),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
