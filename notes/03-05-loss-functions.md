# 03/05 Loss Functions

> The loss is the only definition of "good" the model ever sees. Measure the wrong thing and it gets very good at the wrong thing.

**Done:** 2026-10-22 · Quiz 3/3 · Bridge lesson 8 of 14
**Lesson:** [phases/03-deep-learning-core/05-loss-functions](../phases/03-deep-learning-core/05-loss-functions/)
**My code:** [`labs/03-05-loss-functions/my_losses.py`](../labs/03-05-loss-functions/my_losses.py). I wrote stable BCE-with-logits, the softmax-CE gradient, label-smoothed CE and InfoNCE, and all 6 checks pass.
**Run it:** `python3 labs/03-05-loss-functions/my_losses.py` (no numpy needed)

---

## Remember these 3 things

1. **Classification uses cross-entropy, not MSE.** Through a sigmoid, the gradient with respect to the logit is `p − y` for CE but `2(p − y)·p(1 − p)` for MSE. When the model is confidently wrong, MSE's gradient is about **50× weaker**, so it gets stuck.
2. **Losses take logits, not probabilities.** `sigmoid(−1000)` underflows to 0.0 and `log(0)` is −∞, which turns into NaN gradients. Use `F.binary_cross_entropy_with_logits` and `F.cross_entropy`, and **never put a sigmoid or softmax layer in front of them.**
3. **InfoNCE (contrastive loss) is softmax cross-entropy over similarities.** The candidates are the "classes" and the positive is the correct one. The **negatives** in the denominator are what stop the embeddings collapsing. A collapsed model scores ln(N), the guessing baseline.

---

## In plain English

The optimizer follows the slope of one number. It doesn't know about accuracy, F1, or your business goal. So you have to pick a loss whose slope pushes the model the way you want:

| Task | Loss | What it rewards |
|------|------|-----------------|
| Predict a number (price, latency) | **MSE** `(p − y)²` | Being close. Big errors cost quadratically. |
| Yes / no | **Binary cross-entropy** | High probability on the right answer. Confident mistakes cost hugely. |
| One of K classes (next token!) | **Softmax cross-entropy** `−log p(correct)` | Putting probability mass on the correct class |
| Embeddings (retrieval) | **InfoNCE** | Your true match ranking above every other candidate |
| Rare positives | **Focal loss** `−(1 − p_t)^γ log p_t` | Hard examples. Easy ones are down-weighted. |

**Analogy.** Two teachers marking a student who answers "definitely false" when the truth is "true":
- **The MSE teacher** caps the penalty at 1 point. "Wrong" and "absurdly, confidently wrong" cost about the same, so the student never feels urgency.
- **The cross-entropy teacher** charges about 2.3 more points for every 10× more confidence in the wrong answer, so being confidently wrong is unaffordable.

---

## How loudly each loss shouts (y = 1)

| Model says p = | MSE (p − 1)² | Cross-entropy −log p |
|---|---|---|
| 0.99 (confidently right) | 0.0001 | 0.01 |
| 0.5 (unsure) | 0.25 | 0.69 |
| 0.1 (wrong) | 0.81 | 2.30 |
| 0.01 (confidently wrong) | 0.98 | 4.61 |
| 0.0001 (absurdly wrong) | 0.9998 | 9.21 |

MSE is capped at 1 and flattens out. CE grows without limit.

---

## The gradient story (the real reason)

`p = sigmoid(z)`, and backprop needs dL/dz, chained through the sigmoid damper p(1 − p):

```text
MSE:  dL/dz = 2(p − y) · p(1 − p)          ← multiplied by the damper
CE:   dL/dz = p − y                         ← CE's 1/p cancels the damper exactly
```

**Worked example (y = 1, p = 0.01, confidently wrong):**
- MSE: 2(−0.99)(0.0099) = **−0.0196**
- CE: 0.01 − 1 = **−0.99**
- Compare **magnitudes**, not signed values: |−0.99| is about **50×** |−0.0196|. The sign is the direction, the size is the strength. I compared the signs at first.

**Measured in my code:** both losses trained the same model, starting confidently wrong (w = −6), for 30 steps:

| Loss | Final w | Accuracy |
|------|---------|----------|
| MSE | −5.46 (barely moved) | **0%** |
| BCE | +2.94 (flipped) | **100%** |

**MSE's "learned nothing" number:** on balanced binary data, predicting 0.5 for everything gives MSE = 0.25. If an MSE classifier plateaus there, it learned nothing (the same idea as ln(V) for cross-entropy).

---

## Softmax + cross-entropy (the LLM loss)

```text
loss  = −log softmax(z)[t]
dL/dz = softmax(z) − one_hot(t)
```

**Example:** z = [2, 1, 0.1], t = 0. softmax = [0.659, 0.242, 0.099], so the gradient = **[−0.341, 0.242, 0.099]**. The values always sum to 0.
- Gradient descent subtracts it: the correct logit goes **up**, and each wrong logit goes down in proportion to the probability it wrongly holds.
- This is the update every LLM gets for every training token.

---

## Label smoothing

Target = `(1 − α)·one_hot + α/K`. With α = 0.1 and K = 3: [1, 0, 0] becomes **[0.933, 0.033, 0.033]**.

- **The problem it fixes:** with a hard target, p → 1 needs the correct logit to be infinitely far above the others. Weights grow without limit, and the model ends up overconfident, poorly calibrated, and brittle when the data shifts.
- **With smoothing:** the gradient `p − 0.933` reaches 0 at p = 0.933, which needs only a finite logit gap: ln(0.933 / 0.033) ≈ **3.3**.
- **Measured:** a logit gap of 3.3 gives loss 0.29, and a gap of 50 gives **3.33**. Overconfidence now costs loss.
- `F.cross_entropy(..., label_smoothing=0.1)`. The original Transformer used 0.1.

---

## InfoNCE: how embedding models learn "similar"

```text
InfoNCE = −log( exp(sim(q, pos)/τ) / Σ over all candidates exp(sim(q, c)/τ) )
        = softmax cross-entropy with logits = cosine / τ, correct class = the positive
```

- **Positive:** the matching pair, e.g. a question and the passage that answers it. **Negatives:** every other item in the batch.

**Numbers from my checks** (1 positive + 255 negatives):

| Model | Loss |
|-------|------|
| **Collapsed** (all embeddings identical, every sim = 1, each p = 1/256) | ln 256 = **5.55**, the guessing baseline |
| Trained (sim with the positive 0.9, with negatives 0.1, τ = 0.07) | **0.003** |
| Same trained embeddings at τ = 1.0 | 4.75 (too soft to separate) |

- **The collapsed model has a high loss, not a low one.** sim(q, pos) = 1 looks perfect, but the negatives also score 1, so the denominator punishes it. A loss with no negatives would score collapse *perfectly*. That's why contrastive losses need negatives, and why bigger batches help.
- **Temperature τ:** lower is sharper and separates harder (from 01/06).
- **Batched in PyTorch:** `F.cross_entropy((q @ c.T) / tau, torch.arange(len(q)))`. Row i's positive is column i. This is the loss behind CLIP-style and most retrieval embedding models.

---

## My code

```python
def bce_with_logits(z, y):
    loss = max(z, 0) - z * y + math.log1p(math.exp(-abs(z)))   # stable: finite at z = ±1000
    return loss, sigmoid(z) - y                                 # the simplified gradient

def softmax_ce_grad(logits, target):
    probs = softmax(logits)
    probs[target] -= 1                                          # softmax − one_hot
    return probs

def label_smoothed_ce(logits, target, alpha):
    k = len(logits)
    ls = log_softmax(logits)                                    # compute ONCE (review fix)
    q = [(1 - alpha) * (i == target) + alpha / k for i in range(k)]
    return -sum(qi * ls[i] for i, qi in enumerate(q))

def info_nce(query, positive, negatives, tau):
    logits = [cosine(query, c) / tau for c in [positive] + negatives]
    return -log_softmax(logits)[0]                              # positive is class 0
```

**Review points on what I first wrote:**
1. **`log_softmax` called inside the loop in `label_smoothed_ce`** made it O(K²). With K = 128k vocabulary that's about 16 billion operations per token instead of 128k. Compute it once.
2. In `bce_with_logits` I computed `p = sigmoid(z)`, didn't use it, then called sigmoid again.
3. `math.log(1 + x)` loses the tiny x when x ≈ 1e-18 (1 + x rounds to 1.0). `math.log1p(x)` keeps it.

---

## Where this shows up in AI engineering

- **Fine-tuning dashboards** show softmax cross-entropy averaged over tokens. Know the ln(V) baseline (01/09).
- **Embedding models for RAG** were trained with InfoNCE. A "better embedding model" means it ranks the true passage above more, and harder, negatives. When you fine-tune one, **hard negatives** (similar but wrong passages) matter most.
- **Classifiers and guardrails:** BCE with logits. For rare positives, combine with class weights or focal loss (from 02/17).
- **Distillation:** cross-entropy or KL against a teacher's *soft* probabilities, the same shape as label smoothing (01/09).
- **Silent bugs to recognise in code review:**
  - a sigmoid layer followed by `BCEWithLogits` gives probabilities stuck in (0.5, 0.73) and a loss that plateaus high with no error
  - a softmax followed by `F.cross_entropy`, the same family of bug

## Mistakes to avoid

- Comparing gradients by signed value. **Compare magnitudes.**
- Thinking a collapsed embedding model gets a low InfoNCE loss. It gets ln(N).
- Passing probabilities to a loss that expects logits, or adding a sigmoid or softmax layer before a logits loss.
- `log(sigmoid(z))` or `log(softmax(z))` written by hand. Use the fused, stable versions.
- Recomputing an O(K) quantity inside a loop over K.

## Words

- **Logit:** a raw score before sigmoid or softmax.
- **One-hot:** a target vector with 1 at the correct class and 0 elsewhere.
- **Calibrated:** "90% confident" is right about 90% of the time.
- **Collapse:** an embedding model mapping everything to (nearly) the same vector.
- **Hard negative:** a wrong candidate that looks very similar to the right one.

## Connects to

- **Before:** [03/03 Backpropagation](03-03-backpropagation.md). The sigmoid damper p(1 − p) is exactly what CE cancels.
- **Also:** [01/06](01-06-probability-and-distributions.md) (softmax, temperature, logits into the loss), [01/09](01-09-information-theory.md) (cross-entropy, the ln(V) baseline), [01/14](01-14-norms-and-distances.md) (cosine similarity inside InfoNCE), [02/17](02-17-imbalanced-data.md) (focal loss and class weights).
- **Next:** 05/19 Subword Tokenization. What the "classes" in the LLM's softmax actually are.

---

## Self-test

1. y = 1, p = 0.01. Give the MSE and CE gradients with respect to the logit. Which is stronger, and by about how much?
2. Write dL/dz for softmax cross-entropy. Compute it for z = [2, 1, 0.1], t = 0.
3. Why should a loss take logits instead of probabilities? Give the failing input.
4. What does label smoothing with α = 0.1, K = 3 turn [1, 0, 0] into, and why does that prevent overconfidence?
5. An embedding model collapses, with 1 positive and 255 negatives. What's its InfoNCE loss, and which part of the formula punishes collapse?
6. A model has an `nn.Sigmoid()` layer followed by `BCEWithLogitsLoss`. What do you see, and what's the fix?
7. What does lowering τ in InfoNCE do?

<details>
<summary>Answers</summary>

1. MSE: 2(p − y)p(1 − p) ≈ −0.0196. CE: p − y = −0.99. CE is about 50× stronger, comparing magnitudes.
2. softmax(z) − one_hot(t) = [−0.341, 0.242, 0.099].
3. sigmoid(−1000) underflows to 0.0, and log(0) is −∞, giving inf loss and NaN gradients. Working from the logit keeps the loss at ≈1000, finite.
4. [0.933, 0.033, 0.033]. The gradient reaches zero at p = 0.933, which needs only a finite logit gap (≈ 3.3), so the logits stop growing.
5. ln 256 ≈ 5.55. The negatives in the denominator: they score the same as the positive, so p(positive) = 1/256.
6. Two sigmoids, so the probabilities are stuck in (0.5, 0.731). It can't predict negatives or be confident, and the loss plateaus high with no error. Remove the layer, and apply `torch.sigmoid` only at inference.
7. It sharpens the softmax over similarities, forcing a clearer separation between the positive and the negatives.

</details>
