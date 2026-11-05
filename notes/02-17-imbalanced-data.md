# 02/17 Imbalanced Data

> When 99% of the data is "normal", a model that always says "normal" looks brilliant and is useless.

**Done:** 2026-10-22 · Quiz 3/3 · Bridge lesson 6 of 14
**Lesson:** [phases/02-ml-fundamentals/17-imbalanced-data](../phases/02-ml-fundamentals/17-imbalanced-data/)
**My code:** [`labs/02-17-imbalanced-data/my_imbalanced.py`](../labs/02-17-imbalanced-data/my_imbalanced.py). I wrote class weights, the weighted loss, SMOTE, MCC and best_f1_threshold, and all 8 checks pass.
**Run it:** `/usr/local/bin/python3 labs/02-17-imbalanced-data/my_imbalanced.py` (needs numpy)

---

## Remember these 3 things

1. **There are three levers against imbalance:**
   - change the **loss** (class weights)
   - change the **data** (oversample, undersample, SMOTE)
   - change the **cutoff** (threshold tuning)

   All three move the same decision boundary toward the rare class, so stacking them can overshoot.
2. **Resample only the training data, after the split.** Validation and test must keep the real-world class ratio, or precision and F1 come out inflated.
3. **Judge with MCC, F1 or AUPRC, never accuracy.** MCC uses all four cells of the confusion matrix and gives 0 to a model that always predicts one class.

---

## In plain English

The things worth catching are rare: prompt injections, hallucinations, fraud, disease. A model trained on 99% "safe" examples learns that the cheapest way to be right is to always say "safe". That isn't a bug. It's the model correctly minimising the loss you gave it. So you have to change the game:

| Lever | What it changes | Analogy |
|-------|----------------|---------|
| **Class weights** | Each mistake on the rare class costs more in the loss | A teacher marking the rare questions worth 20 points each |
| **Random oversampling** | Copy rare examples until the classes balance | Photocopying the same 50 practice questions |
| **SMOTE** | Create *new* rare examples between real ones | Writing new practice questions "in between" the existing ones |
| **Undersampling** | Throw away common examples | Studying from fewer easy questions |
| **Threshold tuning** | Move the score cutoff, usually away from 0.5 | Deciding how suspicious is suspicious enough to flag |

---

## Why accuracy fails (worked example)

1,000 transactions: 990 legitimate and 10 fraud. The model always says "legitimate".

| | Predicted fraud | Predicted legit |
|--|---|---|
| **Really fraud** | TP = 0 | FN = 10 |
| **Really legit** | FP = 0 | TN = 990 |

- Accuracy = 990 / 1000 = **99%**, which looks great.
- Recall = 0, F1 = 0 and **MCC = 0**. These show it's worthless.

---

## Lever 1: class weights

**Rule** (the "balanced" weights, which sklearn computes with `class_weight="balanced"`):

```text
w_c = n_samples / (n_classes × n_c)        where n_c = number of samples in class c
```

**Worked example.** 9,900 safe and 100 injections (n = 10,000, 2 classes):
- w_safe = 10,000 / (2 × 9,900) = **0.505**
- w_injection = 10,000 / (2 × 100) = **50**
- Total weight: safe 9,900 × 0.505 = **5,000** and injection 100 × 50 = **5,000**

**Each class now carries exactly half the loss.** That's the whole idea.

**How it enters training.** Multiply each sample's error by its weight:

```python
error = sample_weights * (p - y)       # unweighted logistic regression would be (p - y)
w -= lr * (X.T @ error) / len(y)
b -= lr * np.mean(error)
```

**Effect in my run:** recall at threshold 0.5 went from **0.68 to 1.00**.

---

## Lever 2: SMOTE (Synthetic Minority Oversampling)

**Algorithm.** For each new point:
1. Pick a random minority point **x**.
2. Find its **k nearest minority** neighbours (not majority points, and not x itself).
3. Pick one of those neighbours, **n**, at random.
4. `new = x + t × (n − x)`, with t random in [0, 1). The new point lands somewhere on the segment from x to n.

**Worked example.** x = (1.0, 2.0), n = (1.5, 2.5), t = 0.4:
new = (1.0 + 0.4 × 0.5, 2.0 + 0.4 × 0.5) = **(1.2, 2.2)**.

**Why not just copy points?** Exact copies make the model memorise those few spots, which is overfitting. SMOTE fills in the *region* around them instead.

**Weaknesses:**
- It can create noisy points near the class boundary.
- It ignores where the majority points are.
- It needs at least 2 minority points (k is clamped to m − 1).

My code:

```python
def smote(X_min, k, n_synthetic, rng):
    m = len(X_min)
    k = min(k, m - 1)                                   # can't have more neighbours than other points
    if m < 2 or k == 0:
        raise ValueError("SMOTE requires at least 2 minority samples")
    res = []
    for _ in range(n_synthetic):
        index = rng.integers(0, m)
        point = X_min[index]
        distances = np.linalg.norm(X_min - point, axis=1)   # distances to every minority point at once
        distances[index] = np.inf                            # exclude the point itself
        neighbours = np.argsort(distances)[:k]               # k nearest
        neighbour = neighbours[rng.integers(0, k)]
        t = rng.random()
        res.append(point + t * (X_min[neighbour] - point))
    return np.array(res).reshape(-1, X_min.shape[1])
```

---

## Lever 3: threshold tuning

Sweep the cutoffs on the **validation** set and keep the one with the best F1 (or whatever your metric is).

**Surprise from my run:** after class weighting, the best threshold went **up** to 0.71, not down.
- The weights had pushed the model to over-predict positives: recall was 1.00 at 0.5, but precision was low.
- Raising the threshold won back the precision. Test MCC went from 0.801 at 0.5 to **0.838** at 0.71.

**Lesson:** weighted models aren't calibrated (a score of 0.7 doesn't mean "70% likely"). Always re-tune the threshold after weighting.

My code (with the review fixes applied):

```python
def best_f1_threshold(y_true, probs):
    best_f1, best_threshold = -1.0, None               # initialise. Don't use `'x' in locals()`
    for t in np.unique(probs):                          # distinct values only
        tp, fp, fn, tn = counts(y_true, (probs >= t).astype(int))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1 or (f1 == best_f1 and t > best_threshold):   # ties go to the higher threshold
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1
```

---

## Better metrics

| Metric | Formula | Why it helps |
|--------|---------|--------------|
| F1 | `2PR / (P + R)` | Only looks at the positive class. Ignores TN entirely. |
| F-beta | `(1 + β²)PR / (β²P + R)` | β > 1 favours recall. F2 is common for fraud. |
| **MCC** | `(TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Uses all 4 cells. +1 is perfect, 0 is chance, −1 is inverted. |
| **AUPRC** | Area under the precision–recall curve | **A random model scores the positive rate** (e.g. 0.01), so real gains are visible. AUC-ROC starts at 0.5 whatever the imbalance. |

**MCC worked example.** TP = 8, FP = 2, FN = 2, TN = 988:
(8 × 988 − 2 × 2) / sqrt(10 × 10 × 990 × 990) = 7,900 / 9,900 = **0.798**.

**Where MCC beats F1.** A model that predicts *everything* positive (TP = 10, FP = 990) gets F1 ≈ 0.02, which looks barely alive. Its MCC = **0** (TN + FN = 0 makes the denominator 0, and a constant predictor carries no information).

My code:

```python
def mcc(tp, fp, fn, tn):
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:                  # the model always predicts one class: no information
        return 0.0
    return (tp * tn - fp * fn) / denom
```

---

## The leakage trap: never SMOTE before splitting

A teammate SMOTEs the whole dataset, then splits it and reports great test F1. Two separate things are wrong:

1. **Leakage.** Synthetic training points are interpolated *toward* test points, so the model has partly seen the test set.
2. **Fake prevalence.** The test set is now 50/50 instead of about 1% positive, and precision depends heavily on prevalence. Same model, 90% recall, 2% of the safe examples falsely flagged:

| Test set | Positives / negatives | TP | FP | Precision | F1 |
|----------|----------------------|----|----|-----------|----|
| Real world (1%) | 100 / 9,900 | 90 | 198 | **0.31** | **0.46** |
| After SMOTE (50/50) | 9,900 / 9,900 | 8,910 | 198 | **0.98** | **0.94** |

The model didn't improve. The test set got easier and better balanced. Synthetic points also sit neatly inside the minority cluster, which makes them easier than real attacks.

**Rule:** split first, resample only the training fold, and keep validation and test at real-world ratios. `imblearn.pipeline.Pipeline` enforces this: SMOTE runs during `fit` and is **skipped** at `predict`, and inside `cross_val_score` each fold resamples only its own training part. sklearn's `Pipeline` can't do this, because its steps run on every call and can't change the number of rows or y.

---

## Which approach to use

| Imbalance | Data size | Try |
|-----------|-----------|-----|
| Mild (≈ 80/20) | any | Class weights, then threshold tuning |
| Moderate (≈ 95/5) | 1k to 10k | SMOTE or weights, plus threshold tuning |
| Severe (99/1 or worse) | any | Combine weights, resampling and threshold. Judge with AUPRC and MCC. |
| any | > 10k | Undersampling becomes affordable |
| any | < 1k | Avoid undersampling, because you'd throw away scarce data |

**When you know real costs** (a missed fraud costs $500, a false alarm costs $2), use **cost-sensitive learning**: minimise total cost instead of error count. This is the most principled choice.

---

## Where this shows up in AI engineering

- **Guardrails:** injections, PII leaks and jailbreaks are rare. Report MCC, AUPRC, and precision and recall at the chosen threshold.
- **Hallucination and LLM-judge eval sets:** if 3% of answers are bad, a judge that always says "fine" is 97% accurate.
- **Building eval sets:** keep the test set at production prevalence. Oversampled eval sets inflate precision.
- **Interview line:** "accuracy is wrong here because..., I'd use class weights or SMOTE on the training fold only, tune the threshold on validation, and report AUPRC and MCC."

## Mistakes to avoid

- Accuracy on imbalanced data.
- Resampling before the split, or resampling validation or test (leakage plus fake prevalence).
- Assuming a weighted model's threshold should go **down**. Mine went *up* to 0.71. Tune it, don't guess.
- Tuning the threshold on the test set.
- My code review points:
  - `compute_class_weights` hard-codes the classes 0 and 1 (KeyError if a class is missing, and no multiclass), and calls `np.append` inside a loop, which copies the array every time. A vectorised version: `classes, counts = np.unique(y, return_counts=True); w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}; return np.array([w[v] for v in y])`.
  - In `smote`, I reused the name `index` for the neighbour choice. Use distinct names.
  - In `best_f1_threshold`, I used `'best_f1' not in locals()`, looped over duplicate probabilities, and crashed on empty input. The fixed version is above.

## Words

- **Prevalence / base rate:** the fraction of samples that are positive.
- **Minority / majority class:** the rare and common labels.
- **Calibrated:** a score of 0.7 really means about 70% of such cases are positive.
- **Leakage:** information from validation or test data reaching training.

## Connects to

- **Before:** [02/09 Model Evaluation](02-09-model-evaluation.md), for precision, recall and thresholds.
- **Next:** 03/03 Backpropagation. What the weighted gradient `X.T @ (w * (p - y))` actually is, and how gradients flow through deep networks.

---

## Self-test

1. 9,900 safe and 100 injections. What are the balanced weights, and the total weight per class?
2. Write the SMOTE formula. Why use minority-only neighbours?
3. Name the two things wrong with SMOTE on the full dataset before the split.
4. Compute MCC for TP = 0, FP = 0, FN = 10, TN = 990. Why is that value correct rather than a hack?
5. After class weighting, the tuned threshold rose from 0.5 to 0.71. Explain why.
6. Why prefer AUPRC over AUC-ROC at 1% prevalence?
7. What does `imblearn.pipeline.Pipeline` do at `predict` time that sklearn's can't?

<details>
<summary>Answers</summary>

1. 0.505 and 50. Each class totals 5,000, i.e. half the loss.
2. `new = x + t(n − x)` with t in [0, 1). Neighbours from the majority class would create "minority" points inside majority territory.
3. Leakage (synthetic training points are built from test points), and fake 50/50 test prevalence that inflates precision and F1.
4. The denominator is 0 because TP + FP = 0. Return 0. A constant predictor has no variance, so it can't correlate with the labels: "no information" is the honest score.
5. The weights made the model over-predict positives (recall 1.00, low precision at 0.5). Raising the threshold traded a little recall for much better precision.
6. A random model's AUPRC equals the positive rate (0.01), so gains are visible. AUC-ROC sits at 0.5 for random and is inflated by the mass of true negatives.
7. It skips the sampler. Resampling happens only during `fit`, so validation, test and each CV fold's held-out part stay untouched.

</details>
