# 02/09 Model Evaluation

> Accuracy lies on imbalanced data. Choose the metric that matches the mistake you can't afford.

**Done:** 2026-10-11 · Quiz 3/3 · Bridge lesson 5 of 14
**Lesson:** [phases/02-ml-fundamentals/09-model-evaluation](../phases/02-ml-fundamentals/09-model-evaluation/)
**My code:** [`labs/02-09-model-evaluation/my_metrics.py`](../labs/02-09-model-evaluation/my_metrics.py). I wrote the confusion matrix, P/R/F1, predict_at and best_threshold_for_recall, and all 8 checks pass.
**Run it:** `python3 labs/02-09-model-evaluation/my_metrics.py` (no numpy needed)

---

## Remember these 3 things

1. **Precision = "when I say yes, am I right?" Recall = "of the real yeses, how many did I catch?"** Precision is hurt by false alarms. Recall is hurt by misses.
2. **Raising the threshold gives fewer yeses: precision goes up and recall goes down.** Lowering it does the opposite. Pick the direction based on which mistake is expensive.
3. **Never tune on the test set.** Tune on a validation set, and use the test set once at the end. Otherwise your reported number is fake.

---

## The confusion matrix (everything starts here)

Example: a prompt-injection detector. 1 = injection (block it), 0 = safe.

|              | Model says 1 | Model says 0 |
|--------------|--------------|--------------|
| **Really 1** | **TP**, true positive (caught it) | **FN**, false negative (*missed* it) |
| **Really 0** | **FP**, false positive (*false alarm*) | **TN**, true negative (correctly let it through) |

**How to read the names:** the second letter is what the model *said* (P or N). The first letter is whether it was right (T or F). So "FN" means the model said Negative and was wrong.

---

## The metrics, with a worked example

100 prompts, 10 of them real injections. The model flags 12: 8 real ones and 4 false alarms.
That gives TP = 8, FP = 4, FN = 2 and TN = 86.

| Metric | Formula | Our numbers | Question it answers |
|--------|---------|-------------|---------------------|
| Accuracy | (TP + TN) / all | 94 / 100 = **0.94** | How often am I right overall? |
| Precision | TP / (TP + FP) | 8 / 12 = **0.67** | When I flag something, how often is it real? |
| Recall | TP / (TP + FN) | 8 / 10 = **0.80** | Of the real injections, how many did I catch? |
| F1 | 2PR / (P + R) | **0.73** | A single number balancing P and R |

**The accuracy trap.** On the same data, a model that *never* flags anything gets 90 / 100 = **90% accuracy** while catching nothing (recall = 0). On imbalanced data, accuracy rewards doing nothing.

**Why F1 uses the harmonic mean.** It punishes a lopsided pair. With P = 1.0 and R = 0.1, the arithmetic mean says 0.55, but **F1 = 0.18**. You can't hide a terrible recall behind a perfect precision.

---

## Thresholds: the dial you actually control

Models output a **score** (like 0.73). The threshold turns it into a yes or no: `predict 1 if score ≥ threshold`.

```text
Low threshold  ◄──────────────────────────►  High threshold
 flag lots                                    flag little
 recall ↑  precision ↓                        precision ↑  recall ↓
 (few misses, many false alarms)              (few false alarms, many misses)
```

**Which way to turn it: ask which mistake costs more.**

| System | Expensive mistake | So you want | Threshold |
|--------|-------------------|-------------|-----------|
| Prompt-injection guardrail | Missing an attack (FN) | High recall | **Lower** |
| 3 a.m. on-call pager | Waking someone for nothing (FP) | High precision | **Higher** |
| Cancer screening (first pass) | Missing cancer (FN) | High recall | Lower |
| Auto-deleting "spam" emails | Deleting a real email (FP) | High precision | Higher |

In the lesson I said to lower the threshold for the pager too. That's wrong: the pager hates false alarms, so you raise it.

**The production pattern.** Recall must be at least X, and then you want the fewest false alarms. Among the thresholds with recall ≥ X, pick the one with the best precision. That's my `best_threshold_for_recall`.

---

## Same idea for RAG retrieval

The retriever returns the top k chunks. Some of them are relevant.

- **precision@k** = relevant chunks in the top k ÷ k ("how much of what I retrieved is useful?")
- **recall@k** = relevant chunks in the top k ÷ all relevant chunks ("did I find what I needed?")

**Example.** Top 5 results contain 2 of the 3 relevant chunks. P@5 = 2/5 = **0.4** and R@5 = 2/3 = **0.67**.

**A bigger k works like a lower threshold:** recall goes up and precision goes down. The extra junk chunks dilute the LLM's context, which is why you retrieve many and then **rerank** down to a few.

---

## My code

```python
def best_threshold_for_recall(y_true, scores, min_recall):
    best = None
    best_prec = 0.0
    best_recall = 0.0
    for threshold in set(scores):                     # try each distinct score as the cutoff
        pred = predict_at(scores, threshold)
        rec = recall(y_true, pred)
        prec = precision(y_true, pred)
        if rec >= min_recall:                         # 1. must meet the recall floor
            if best is None or prec > best_prec or (prec == best_prec and threshold > best):
                best, best_prec, best_recall = threshold, prec, rec   # 2. then best precision (ties: higher threshold)
    return (best, best_prec, best_recall) if best is not None else None
```

**Review points** (things an interviewer would notice):

1. **Magic indices.** My `confusion_matrix` returns a list, and I read it with `m[0]`, `m[2]` and so on. Unpack it instead: `tp, fp, fn, tn = confusion_matrix(...)`.
2. **String labels fail silently.** `"1" == 1` is False, so string labels match neither branch and every metric comes out 0.0. Guard with `assert set(y_true) <= {0, 1}`.
3. **O(n²) sweep.** Re-scoring all n samples for each of n thresholds is quadratic. A faster way is to sort by score once and walk down, updating TP and FP counts as you go, which is O(n log n). sklearn's `precision_recall_curve` does this.

---

## Covered only briefly

- **Train / validation / test:** train on train, tune choices (thresholds, hyperparameters) on validation, and report on test **once**.
- **K-fold cross-validation:** split the data into K parts. Each part is the validation set exactly once and training data K−1 times. Average the K scores. Useful when the data is small.
- **Learning curves:**
  - train high, validation low, with a gap that doesn't close: **overfitting** (high variance). Fix with a simpler model or more regularisation.
  - both low: **underfitting** (high bias). Fix with a bigger model or better features.
- **AUC-ROC:** the probability that a random positive gets a higher score than a random negative. It doesn't depend on any one threshold. 0.5 is random, 1.0 is perfect, and below 0.5 means the scores are flipped. *On very imbalanced data, prefer AUPRC.* See [02/17](02-17-imbalanced-data.md).

## Where this shows up in AI engineering

- **Guardrails** (injection, PII, toxicity) are binary classifiers with a threshold. Report precision and recall, not accuracy.
- **LLM-as-judge:** measure the judge's precision and recall against human labels before you trust it.
- **RAG eval:** precision@k and recall@k for the retriever.
- **Interviews:** "Which metric would you use, and why?" Answer by naming the expensive mistake first.

## Mistakes to avoid

- Reporting accuracy on imbalanced data.
- Turning the threshold the wrong way for the cost (my pager mix-up).
- Tuning a threshold on the test set.
- Magic list indices, silent string labels, and an O(n²) sweep (see the review points above).

## Connects to

- **Before:** [01/09 Information Theory](01-09-information-theory.md)
- **Next:** [02/17 Imbalanced Data](02-17-imbalanced-data.md). What to do when positives are rare: class weights, SMOTE, MCC and threshold tuning.

---

## Self-test

1. A spam filter scores 99% accuracy on mail that's 1% spam. Is it good?
2. Define precision and recall in one plain sentence each.
3. TP = 6, FP = 6, FN = 4. What are precision, recall and F1?
4. You raise the threshold from 0.5 to 0.8. What happens to precision and recall?
5. A hallucination detector must catch at least 95% of hallucinations. Which metric is the constraint, and which do you then maximise?
6. The top 10 chunks contain 3 of the 4 relevant ones. What are P@10 and R@10?
7. Train is 0.95 and validation is 0.60, and more data doesn't help. Diagnosis and fix?

<details>
<summary>Answers</summary>

1. You can't tell. Always saying "not spam" also gets 99%. Check precision and recall.
2. Precision: when it says yes, how often is it right? Recall: of all the real yeses, how many did it find?
3. P = 6/12 = 0.5, R = 6/10 = 0.6, F1 = 2(0.5)(0.6)/1.1 ≈ 0.545.
4. Precision usually goes up. Recall goes down or stays the same, and never goes up.
5. Recall ≥ 0.95 is the constraint. Among the thresholds that meet it, pick the one with the highest precision.
6. P@10 = 0.3 and R@10 = 0.75.
7. Overfitting (high variance). Use a simpler model, regularisation or dropout.

</details>
