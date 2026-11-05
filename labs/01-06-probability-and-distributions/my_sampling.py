# Bridge lesson 3: Probability and Distributions. Build the logits -> token pipeline.
# Run: python3 labs/01-06-probability-and-distributions/my_sampling.py   (stdlib only, no numpy)
import math
import random


def softmax(logits, temperature=1.0):
    # logits: list of floats. Return a list of probabilities that sums to 1.
    # Requirements:
    #   - divide by temperature first; raise ValueError if temperature <= 0
    #   - numerically stable: [1000, 999, 998] must not overflow (hint: shift invariance)
    # logits = [1.0, 2.0, 3.0]
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled_logits = []
    for logit in logits:
        logit = float(logit / temperature)
        scaled_logits.append(logit)
    
    max_logit = max(scaled_logits)

    for i in range(len(scaled_logits)):
        scaled_logits[i] = float(scaled_logits[i] - max_logit)
    
    denominator = sum(math.exp(l) for l in scaled_logits)
    softmax = []
    for i in range(len(scaled_logits)):
        softmax.append(float((math.exp(scaled_logits[i]))/denominator))
    return softmax


def log_softmax(logits):
    # Return log-probabilities directly, WITHOUT computing softmax and then taking log.
    # It must work for [1000.0, 0.0]: softmax underflows the second prob to 0.0, and log(0) crashes.
    # Hint: log softmax(z)_i = z_i - log(sum_j exp(z_j)). Make that log-sum-exp stable.
    log_softmax = []
    max_logit = max(logits)
    logsum = max_logit + math.log(sum(math.exp(l - max_logit) for l in logits))
    for i in range(len(logits)):
        log_softmax.append(logits[i] - logsum)
    return log_softmax


def cross_entropy(logits, target_index):
    # The loss when the correct class/token is target_index. One line, using log_softmax.
    return -(log_softmax(logits)[target_index])


def sample_categorical(probs, rng):
    # Draw ONE index i with probability probs[i]. Use rng.random() (uniform in [0, 1)).
    # Hint: walk the running (cumulative) sum until it passes the random number.
    r = rng.random()
    cumulative = 0.0
    for i, val in enumerate(probs):
        cumulative+= val
        if cumulative >= r:
            return i
    return len(probs) -1

# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def close(a, b, tol=1e-3):
    return all(abs(x - y) < tol for x, y in zip(a, b)) and len(a) == len(b)


def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_basic():
    p = softmax([2.0, 1.0, 0.1])
    return close(p, [0.659, 0.242, 0.099]) and abs(sum(p) - 1) < 1e-9, [round(x, 3) for x in p]


def t_shift():
    a, b = softmax([2.0, 1.0, 0.1]), softmax([102.0, 101.0, 100.1])
    return close(a, b, 1e-9), ""


def t_overflow():
    p = softmax([1000.0, 999.0, 998.0])
    return all(math.isfinite(x) for x in p) and close(p, [0.665, 0.245, 0.090]), [round(x, 3) for x in p]


def t_temperature():
    cold, hot = softmax([2.0, 1.0, 0.1], 0.1), softmax([2.0, 1.0, 0.1], 10.0)
    return cold[0] > 0.999 and max(hot) - min(hot) < 0.1, f"T=0.1 top={cold[0]:.5f}, T=10 spread={max(hot) - min(hot):.3f}"


def t_bad_temperature():
    try:
        softmax([1.0, 2.0], 0)
    except ValueError:
        return True, ""
    return False, "expected ValueError for temperature=0"


def t_log_softmax():
    z = [2.0, 1.0, 0.1]
    ok1 = close(log_softmax(z), [math.log(p) for p in softmax(z)], 1e-9)
    ls = log_softmax([1000.0, 0.0])
    ok2 = all(math.isfinite(x) for x in ls) and abs(ls[1] - (-1000.0)) < 1e-6
    return ok1 and ok2, f"log_softmax([1000, 0]) = {ls}"


def t_cross_entropy():
    ce = cross_entropy([2.0, 1.0, 0.1], 0)
    return abs(ce - 0.4170) < 1e-3, f"{ce:.4f}"


def t_sampling():
    rng, probs, n = random.Random(0), [0.659, 0.242, 0.099], 100_000
    counts = [0, 0, 0]
    for _ in range(n):
        counts[sample_categorical(probs, rng)] += 1
    freq = [c / n for c in counts]
    sure = {sample_categorical([0.0, 1.0, 0.0], rng) for _ in range(1000)}
    return close(freq, probs, 0.01) and sure == {1}, f"freq={[round(f, 3) for f in freq]}"


if __name__ == "__main__":
    tests = [
        ("softmax matches [0.659, 0.242, 0.099] and sums to 1", t_basic),
        ("softmax is shift-invariant (+100 changes nothing)", t_shift),
        ("softmax survives [1000, 999, 998] without overflow", t_overflow),
        ("temperature: T=0.1 ~ argmax, T=10 ~ flat", t_temperature),
        ("temperature <= 0 raises ValueError", t_bad_temperature),
        ("log_softmax equals log(softmax) and survives [1000, 0]", t_log_softmax),
        ("cross_entropy([2, 1, 0.1], target=0) = 0.4170", t_cross_entropy),
        ("sampling frequencies match probs over 100k draws", t_sampling),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
