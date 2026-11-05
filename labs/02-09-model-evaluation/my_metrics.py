# Bridge lesson 5: Model Evaluation. Build classification metrics and a threshold picker.
# Run: python3 labs/02-09-model-evaluation/my_metrics.py   (stdlib only)
# Labels are 1 (positive, e.g. "injection -> block") and 0 (negative).


def confusion_matrix(y_true, y_pred):
    # Return (tp, fp, fn, tn) counts.
    matrix = [0,0,0,0]
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 0:
                matrix[2] +=1
            else:
                matrix[0] += 1
        
        if y_true[i] == 0:
            if y_pred[i] == 0:
                matrix[3] +=1
            else:
                matrix[1] +=1 
    return matrix


def accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true,y_pred)
    if (matrix[0] + matrix[1] + matrix[2] + matrix[3]) == 0:
        return 0.0
    return (matrix[0]+matrix[3])/(matrix[0] + matrix[1] + matrix[2] + matrix[3])


def precision(y_true, y_pred):
    # TP / (TP + FP). If nothing was predicted positive, return 0.0 (don't crash).
    matrix = confusion_matrix(y_true,y_pred)
    if (matrix[0] + matrix[1]) == 0:
        return 0.0
    return matrix[0]/(matrix[0] + matrix[1])
    


def recall(y_true, y_pred):
    # TP / (TP + FN). If there are no actual positives, return 0.0.
    matrix = confusion_matrix(y_true,y_pred)
    if (matrix[0] + matrix[2]) == 0:
        return 0.0
    return matrix[0]/(matrix[0] + matrix[2])


def f1(y_true, y_pred):
    # Harmonic mean of precision and recall. 0.0 if both are 0.
    
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec == 0.0 and rec == 0.0:
        return 0.0 
    
    return (2*prec*rec)/(prec+rec)


def predict_at(scores, threshold):
    # 1 if score >= threshold else 0, for each score.
    res = []
    for score in scores:
        if score >= threshold:
            res.append(1)
        else:
            res.append(0)
    return res


def best_threshold_for_recall(y_true, scores, min_recall):
    # The production question: "We must catch at least min_recall of injections.
    # Which threshold gives us that with the FEWEST false alarms?"
    # Try each distinct score as a threshold. Among thresholds whose recall >= min_recall,
    # return the one with the highest precision (ties: the higher threshold).
    # Return (threshold, precision, recall). If none qualifies, return None.
    best = None
    best_prec = 0.0
    best_recall = 0.0
    for threshold in set(scores):
        pred = predict_at(scores, threshold)
        rec = recall(y_true, pred)
        prec = precision(y_true, pred)
        if rec >= min_recall:
            if best is None or prec > best_prec or (prec == best_prec and threshold > best):
                best = threshold
                best_prec = prec
                best_recall = rec
    if best is not None:
        return (best, best_prec, best_recall)
    else:
        return None


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

Y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
S = [0.95, 0.80, 0.60, 0.30, 0.70, 0.40, 0.20, 0.10, 0.05, 0.50]


def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def near(a, b):
    return abs(a - b) < 1e-9


def t_confusion():
    cm = confusion_matrix(Y, predict_at(S, 0.5))
    return tuple(cm) == (3, 2, 1, 4), f"(tp, fp, fn, tn) = {tuple(cm)}"


def t_metrics_05():
    p = predict_at(S, 0.5)
    vals = accuracy(Y, p), precision(Y, p), recall(Y, p), f1(Y, p)
    return near(vals[0], 0.7) and near(vals[1], 0.6) and near(vals[2], 0.75) and near(vals[3], 2 / 3), \
        "acc={:.3f} P={:.3f} R={:.3f} F1={:.3f}".format(*vals)


def t_threshold_tradeoff():
    lo, hi = predict_at(S, 0.3), predict_at(S, 0.9)
    ok = precision(Y, hi) > precision(Y, lo) and recall(Y, hi) < recall(Y, lo)
    return ok, f"t=0.3: P={precision(Y, lo):.3f} R={recall(Y, lo):.3f} | t=0.9: P={precision(Y, hi):.3f} R={recall(Y, hi):.3f}"


def t_zero_division():
    none_pred = [0] * len(Y)
    no_pos = [0] * len(Y)
    ok = precision(Y, none_pred) == 0.0 and recall(no_pos, none_pred) == 0.0 and f1(Y, none_pred) == 0.0
    return ok, ""


def t_accuracy_trap():
    y = [1] * 5 + [0] * 95
    lazy = [0] * 100
    a, r, f = accuracy(y, lazy), recall(y, lazy), f1(y, lazy)
    return near(a, 0.95) and r == 0.0 and f == 0.0, f"always-no model: accuracy={a:.2f}, recall={r:.2f}, F1={f:.2f}"


def t_harmonic():
    y = [1] * 10 + [0] * 10
    pred = [1] + [0] * 9 + [0] * 10           # P = 1.0, R = 0.1
    return abs(f1(y, pred) - 0.1818) < 1e-3, f"P=1.0, R=0.1 -> F1={f1(y, pred):.4f} (not 0.55)"


def t_best_threshold():
    full = best_threshold_for_recall(Y, S, 1.0)
    most = best_threshold_for_recall(Y, S, 0.75)
    ok = full is not None and most is not None \
        and near(full[0], 0.30) and near(full[1], 4 / 7) and near(full[2], 1.0) \
        and near(most[0], 0.60) and near(most[1], 0.75) and near(most[2], 0.75)
    return ok, f"recall>=1.0 -> {full}; recall>=0.75 -> {most}"


def t_best_threshold_none():
    return best_threshold_for_recall([0, 0, 0], [0.1, 0.5, 0.9], 0.5) is None, "no positives -> None"


if __name__ == "__main__":
    tests = [
        ("confusion matrix at t=0.5 is (3, 2, 1, 4)", t_confusion),
        ("accuracy / precision / recall / F1 at t=0.5", t_metrics_05),
        ("raising threshold: precision up, recall down", t_threshold_tradeoff),
        ("no division-by-zero crashes", t_zero_division),
        ("accuracy trap: 95% accuracy, 0 recall", t_accuracy_trap),
        ("F1 is a harmonic mean (punishes imbalance)", t_harmonic),
        ("best_threshold_for_recall picks the right cutoff", t_best_threshold),
        ("best_threshold_for_recall returns None when impossible", t_best_threshold_none),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
