# Bridge lesson 8: Loss Functions. Build the losses behind classifiers, LLMs and embedding models.
# Run: python3 labs/03-05-loss-functions/my_losses.py   (stdlib only, no numpy)
# Every gradient here is with respect to the LOGIT z (the raw score before sigmoid/softmax).
import math
import random


# Provided helpers (you wrote these in 01/06 and 01/14).
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))


def log_softmax(z):
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return [v - lse for v in z]


def softmax(z):
    return [math.exp(v) for v in log_softmax(z)]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def bce_with_logits(z, y):
    # Binary cross-entropy computed from the LOGIT z, not from p (this is PyTorch's BCEWithLogitsLoss).
    # Return (loss, dloss_dz).
    #   loss = -(y*log(p) + (1-y)*log(1-p)) with p = sigmoid(z), but computing it that way gives
    #   log(0) = -inf for z = 1000 or z = -1000. It must stay finite. A stable rewrite:
    #       loss = max(z, 0) - z*y + log(1 + exp(-|z|))
    #   dloss_dz: the simplified gradient from the lesson (one short expression).
    p = sigmoid(z)
    loss = max(z,0) - z*y + math.log(1+math.exp(-abs(z)))
    dloss_dz = p - y
    return (loss,dloss_dz)

def softmax_ce_grad(logits, target):
    # Gradient of  -log softmax(logits)[target]  with respect to every logit.
    # Return a list the same length as logits.
    probs = softmax(logits)
    probs[target]-= 1
    return probs


def label_smoothed_ce(logits, target, alpha):
    # Cross-entropy against the smoothed target  q = (1 - alpha) * one_hot + alpha / K,  K = len(logits).
    #   loss = -sum_i q_i * log_softmax(logits)_i
    # alpha = 0 must give exactly the ordinary cross-entropy -log_softmax(logits)[target].
    k = len(logits)
    one_hot = [0] * k
    one_hot[target] = 1
    smoothed = [(1-alpha) * value + (alpha/k) for value in one_hot]
    ls=log_softmax(logits)
    return -sum(q * ls[i] for i, q in enumerate(smoothed))


    


def info_nce(query, positive, negatives, tau):
    # Contrastive loss: softmax cross-entropy where the "classes" are the candidates
    # [positive] + negatives, the correct class is the positive, and the logits are cosine(query, c) / tau.
    # Must be stable for tau = 0.01 (logits up to 100). Hint: log_softmax already is.
    sim_positive = cosine(query, positive) / tau
    sim_negatives = [cosine(query, neg) / tau for neg in negatives]
    logits = [sim_positive] + sim_negatives
    return -log_softmax(logits)[0]


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def numeric(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_bce_values():
    naive = lambda z, y: -(y * math.log(sigmoid(z)) + (1 - y) * math.log(1 - sigmoid(z)))
    ok = all(abs(bce_with_logits(z, y)[0] - naive(z, y)) < 1e-9 for z in (-3.0, -0.4, 0.0, 1.7, 4.0) for y in (0.0, 1.0))
    big = bce_with_logits(1000.0, 0.0)[0], bce_with_logits(-1000.0, 1.0)[0], bce_with_logits(1000.0, 1.0)[0]
    ok = ok and abs(big[0] - 1000) < 1e-9 and abs(big[1] - 1000) < 1e-9 and big[2] < 1e-12
    return ok, f"matches the naive formula; z=+1000,y=0 -> {big[0]:.1f}, z=-1000,y=1 -> {big[1]:.1f}, z=+1000,y=1 -> {big[2]:.1e}"


def t_bce_grad():
    worst = max(abs(bce_with_logits(z, y)[1] - numeric(lambda t: bce_with_logits(t, y)[0], z))
                for z in (-3.0, -0.4, 0.0, 1.7, 4.0) for y in (0.0, 1.0))
    z = math.log(0.01 / 0.99)                       # p = 0.01: confidently wrong for y = 1
    g_bce = bce_with_logits(z, 1.0)[1]
    g_mse = 2 * (sigmoid(z) - 1.0) * sigmoid(z) * (1 - sigmoid(z))
    return worst < 1e-6 and abs(g_bce + 0.99) < 1e-9, \
        f"vs numeric max err {worst:.1e}; at p=0.01,y=1: |BCE grad| {abs(g_bce):.3f} vs |MSE grad| {abs(g_mse):.4f} ({abs(g_bce / g_mse):.0f}x)"


def t_softmax_grad():
    g = softmax_ce_grad([2.0, 1.0, 0.1], 0)
    ok = all(abs(a - b) < 1e-3 for a, b in zip(g, [-0.341, 0.242, 0.099])) and abs(sum(g)) < 1e-12
    rng = random.Random(3)
    z = [rng.uniform(-3, 3) for _ in range(6)]
    g = softmax_ce_grad(z, 4)
    for i in range(6):
        f = lambda v, i=i: -log_softmax(z[:i] + [v] + z[i + 1:])[4]
        ok = ok and abs(g[i] - numeric(f, z[i])) < 1e-6
    return ok, f"[2,1,0.1], target 0 -> {[round(v, 3) for v in softmax_ce_grad([2.0, 1.0, 0.1], 0)]}; sums to 0; matches numeric"


def t_label_smoothing():
    z = [2.0, 1.0, 0.1]
    plain = -log_softmax(z)[0]
    ok = abs(label_smoothed_ce(z, 0, 0.0) - plain) < 1e-12 and abs(label_smoothed_ce(z, 0, 0.1) - 0.5137) < 1e-3
    gap_ok = label_smoothed_ce([3.3, 0.0, 0.0], 0, 0.1)
    gap_huge = label_smoothed_ce([50.0, 0.0, 0.0], 0, 0.1)
    ok = ok and gap_huge > gap_ok and -log_softmax([50.0, 0.0, 0.0])[0] < -log_softmax([3.3, 0.0, 0.0])[0]
    return ok, f"alpha=0 equals plain CE; logit gap 3.3 -> {gap_ok:.3f}, gap 50 -> {gap_huge:.3f} (overconfidence now COSTS loss)"


def t_info_nce():
    rng = random.Random(0)
    v = [rng.gauss(0, 1) for _ in range(8)]
    collapsed = info_nce(v, v, [v] * 255, 0.07)
    q = [1.0, 0.0, 0.0]
    pos = [0.9, 0.43589, 0.0]                         # cosine 0.9 with q
    neg = [0.1, 0.0, 0.99499]                         # cosine 0.1 with q
    good = info_nce(q, pos, [neg] * 255, 0.07)
    hot = info_nce(q, pos, [neg] * 255, 1.0)
    tiny = info_nce(q, pos, [neg] * 255, 0.01)
    ok = abs(collapsed - math.log(256)) < 1e-9 and good < 0.01 and hot > good and math.isfinite(tiny)
    return ok, f"collapsed -> {collapsed:.3f} (= ln 256), trained@tau=0.07 -> {good:.4f}, same model@tau=1.0 -> {hot:.3f}"


def t_training():
    # 1-D data: positives at x > 0. Start CONFIDENTLY WRONG (w = -6) and train both losses the same way.
    xs = [x / 10 for x in range(-20, 21) if x != 0]
    ys = [1.0 if x > 0 else 0.0 for x in xs]
    results = {}
    for name in ("mse", "bce"):
        w, b = -6.0, 0.0
        for _ in range(30):
            gw = gb = 0.0
            for x, y in zip(xs, ys):
                z = w * x + b
                if name == "bce":
                    g = bce_with_logits(z, y)[1]
                else:
                    p = sigmoid(z)
                    g = 2 * (p - y) * p * (1 - p)
                gw += g * x / len(xs)
                gb += g / len(xs)
            w -= 1.0 * gw
            b -= 1.0 * gb
        acc = sum((sigmoid(w * x + b) > 0.5) == (y == 1.0) for x, y in zip(xs, ys)) / len(xs)
        results[name] = (w, acc)
    ok = results["bce"][1] == 1.0 and results["mse"][1] < 0.5
    return ok, f"after 30 steps from w=-6: MSE w={results['mse'][0]:.2f} acc={results['mse'][1]:.0%}, BCE w={results['bce'][0]:.2f} acc={results['bce'][1]:.0%}"


if __name__ == "__main__":
    tests = [
        ("bce_with_logits: correct and finite at z = +/-1000", t_bce_values),
        ("bce_with_logits gradient: numeric match, ~50x MSE when confidently wrong", t_bce_grad),
        ("softmax CE gradient = softmax - one_hot", t_softmax_grad),
        ("label smoothing: alpha=0 is plain CE, huge logit gaps now cost loss", t_label_smoothing),
        ("InfoNCE: collapse = ln 256, trained model ~0, temperature matters", t_info_nce),
        ("training from confidently wrong: BCE recovers, MSE stays stuck", t_training),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
