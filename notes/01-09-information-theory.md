# 01/09 Information Theory

> Loss, perplexity and logprobs are the same number seen from different angles: how surprised the model was.

**Done:** 2026-10-11 · Quiz 3/3 · Bridge lesson 4 of 14
**Lesson:** [phases/01-math-foundations/09-information-theory](../phases/01-math-foundations/09-information-theory/)
**My code:** [`labs/01-09-information-theory/my_info.py`](../labs/01-09-information-theory/my_info.py). I wrote entropy, cross_entropy, kl_divergence and sequence_perplexity, and all 8 checks pass.
**Run it:** `python3 labs/01-09-information-theory/my_info.py` (no numpy needed)

---

## Remember these 3 things

1. **Surprise = −log p.** A likely event (p near 1) is barely surprising. A rare one (p near 0) is very surprising. Every formula below is an *average surprise*.
2. **LLM training loss is cross-entropy, and perplexity = exp(loss).** A perplexity of 8 means "as unsure as picking uniformly among 8 tokens".
3. **Sanity number: a loss of about ln(vocab size)** (≈ 10.8 for a 50k vocabulary) means the model is guessing uniformly. If a fine-tune *starts* there, something is broken.

---

## In plain English: start with surprise

You're told the result of an event that had probability p. How surprised are you?

| Event | p | Surprise −log₂ p (bits) |
|-------|---|------------------------|
| Fair coin lands heads | 0.5 | 1 bit |
| Rolling a specific number on a 4-sided die | 0.25 | 2 bits |
| A 1-in-1024 event | 1/1024 | 10 bits |
| A certain event | 1 | 0 bits |

Using log₂ gives **bits**. Using the natural log (ln) gives **nats**, which is what PyTorch uses. Same idea, different unit.

From there:

- **Entropy H(P):** the average surprise when outcomes really follow P. It measures how unpredictable P itself is.
- **Cross-entropy H(P, Q):** the average surprise when outcomes follow P but **you predicted Q**. If your guess is wrong, you're more surprised.
- **KL divergence:** the *extra* surprise caused by using Q instead of the true P. KL = cross-entropy − entropy.

**Analogy.** A weather forecaster.
- Entropy is how unpredictable the weather really is.
- Cross-entropy is how surprised you are by the weather if you trust *this* forecaster.
- KL is how much worse this forecaster is than a perfect one.

---

## Worked examples

**Entropy:**
- Fair coin [0.5, 0.5]: H = **1 bit**, the maximum for two outcomes.
- Biased coin [0.99, 0.01]: H ≈ **0.08 bits**, because you almost always know what's coming.
- 4 equal outcomes: H = **2 bits**.
- [0.7, 0.2, 0.1]: H ≈ 1.16 bits.

**Uniform is the most uncertain.** The more lopsided the distribution, the lower its entropy.

**Cross-entropy and KL.** The truth is P = [0.5, 0.5] (a fair coin) and the model says Q = [0.9, 0.1]:
- H(P, Q) = −(0.5 log₂ 0.9 + 0.5 log₂ 0.1) ≈ **1.74 bits**
- H(P) = 1 bit
- **KL(P‖Q) = 1.74 − 1 = 0.74 bits** of wasted surprise
- Swap them and you get KL(Q‖P) ≈ 0.53 bits. **Not the same, so KL is not symmetric** and isn't a true distance.

**Perplexity of a sentence.** The model gave the 3 actual tokens probabilities 0.5, 0.25 and 0.125:
- The average negative log-probability is (0.693 + 1.386 + 2.079) / 3 = 1.386 nats.
- Perplexity = e^1.386 = **4**, so the model was "choosing among about 4 options" per token.

---

## Formulas cheat sheet

`p` is the true probability and `q` is the model's predicted probability.

| Name | Formula | In words |
|------|---------|----------|
| Surprise | `−log p` | How unexpected one outcome is |
| Entropy | `H(P) = −Σ p log p` | Average surprise under the truth |
| Cross-entropy | `H(P, Q) = −Σ p log q` | Average surprise using the model's guesses |
| KL divergence | `KL(P‖Q) = H(P, Q) − H(P)` | Extra cost of the wrong model. Always ≥ 0. |
| Perplexity | `exp(−average log q(actual token))` | Effective number of choices per token |
| Mutual information | `I(X; Y) = H(X) − H(X | Y)` | How much knowing Y reduces uncertainty about X |

**Why training on cross-entropy works.** The data's entropy H(P) is a fixed number you can't change. So minimising cross-entropy is the same as minimising KL, which pushes the model's distribution Q toward the true one P.

**With a one-hot target** (only the correct token has p = 1), the sum collapses to one term: `loss = −log q(correct token)`. That's exactly the cross-entropy from [01/06](01-06-probability-and-distributions.md).

---

## The LLM numbers to know

| Situation | Loss (nats) | Perplexity | Meaning |
|-----------|-------------|-----------|---------|
| Uniform guess, 32k vocabulary | ln 32000 ≈ 10.4 | 32,000 | Knows nothing |
| Uniform guess, 50k vocabulary | ln 50000 ≈ **10.8** | 50,000 | Knows nothing |
| Uniform guess, 128k vocabulary | ln 128000 ≈ 11.8 | 128,000 | Knows nothing |
| A pretrained model at the start of fine-tuning | roughly 2 to 3 | roughly 7 to 20 | Normal |

**If your first fine-tuning step logs about 10.8,** don't read it as "not trained yet". A pretrained model should already be far below that. Look for a broken setup: wrong tokenizer, misaligned labels, or weights that didn't load. I first misread this in the lesson.

---

## My code

```python
def entropy(p, base=math.e):
    ent = 0
    for i in p:
        if i != 0:                      # 0 × log 0 is treated as 0 (it never happens)
            ent += i * math.log(i, base)
    return -ent

def cross_entropy(p, q, base=math.e):
    cross = 0
    for i in range(len(p)):
        if p[i] != 0:
            if q[i] == 0:
                return math.inf         # the model said "impossible" and it happened: infinite surprise
            cross += p[i] * math.log(q[i], base)
    return -cross

def kl_divergence(p, q, base=math.e):
    return cross_entropy(p, q) - entropy(p)        # ⚠ BUG: should pass base to both calls

def sequence_perplexity(token_logprobs):
    avg = 0
    for lp in token_logprobs:
        avg += lp
    return math.exp(-avg / len(token_logprobs))    # ⚠ crashes on an empty list
```

**Fixes I still need to make in my file** (from the review):

```python
def kl_divergence(p, q, base=math.e):
    return cross_entropy(p, q, base) - entropy(p, base)

def sequence_perplexity(token_logprobs):
    if not token_logprobs:
        raise ValueError("empty sequence")
    return math.exp(-sum(token_logprobs) / len(token_logprobs))
```

---

## Where this shows up in AI engineering

- **Fine-tuning dashboards** show the training and validation loss (cross-entropy). Know the uniform baseline for your vocabulary size.
- **API `logprobs`** give the log-probability of each generated token. Average them for a confidence score, or `exp(−mean)` for perplexity. Use it to flag low-confidence answers for review.
- **KL penalty in RLHF/DPO:** keeps the fine-tuned model from drifting too far from the base model.
- **Distillation:** a small model is trained to match a big model's distribution by minimising KL.

## Mistakes to avoid

- **Low entropy means confident, not correct.** A hallucinating model can be very sure of itself.
- **Low perplexity means fluent and well-fitted, not factual.**
- **p > 0 but q = 0 gives infinite cross-entropy.** That's why probabilities get smoothed or clipped.
- **KL is not symmetric.** KL(P‖Q) ≠ KL(Q‖P).
- **PMF vs PDF:** I mixed these up in a warm-up. A PMF covers discrete outcomes and sums to 1. A PDF covers continuous values, integrates to 1, and can be above 1 at a point.

## Words

- **Bit / nat:** units of information (log₂ vs ln). 1 nat ≈ 1.44 bits.
- **One-hot:** a vector that is 1 at the correct index and 0 everywhere else.
- **Logprob:** the log of a probability. Always ≤ 0. Closer to 0 means more confident.

## Connects to

- **Before:** [01/06 Probability and Distributions](01-06-probability-and-distributions.md), where softmax and cross-entropy first appear.
- **Next:** [02/09 Model Evaluation](02-09-model-evaluation.md).

---

## Self-test

1. A fair coin vs a 99/1 coin: which has more entropy, and roughly how much in bits?
2. Write KL in terms of cross-entropy and entropy. Why can't it be negative?
3. A model gives the actual tokens probabilities 0.5, 0.5, 0.5, 0.5. What is the perplexity?
4. The first fine-tuning step logs loss 10.8 with a 50k vocabulary. What do you conclude?
5. Model A has perplexity 3 and model B has 12 on your eval set. Is A more truthful?
6. Why is minimising cross-entropy the same as minimising KL?

<details>
<summary>Answers</summary>

1. The fair coin, with 1 bit. The 99/1 coin has about 0.08 bits.
2. KL = H(P, Q) − H(P). Cross-entropy is never smaller than entropy (Gibbs' inequality), and the two are equal only when Q = P.
3. Each surprise is ln 2, so perplexity = e^(ln 2) = **2**.
4. 10.8 = ln(50000), the uniform-guess baseline. A pretrained model shouldn't be there, so the setup is broken (tokenizer, labels or weights).
5. Not necessarily. A fits the text better and is more confident, but perplexity says nothing about facts.
6. KL = cross-entropy − H(P), and H(P) of the data is a constant, so the two have the same minimiser.

</details>
