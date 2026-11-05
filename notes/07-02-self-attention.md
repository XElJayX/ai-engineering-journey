# 07/02 Self-Attention from Scratch

> Every token asks "who in this sequence matters to me?" and blends in their information. Where it looks is learned, not hard-coded.

**Done:** 2026-10-26 · Quiz 3/3 · Bridge lesson 12 of 14
**Lesson:** [phases/07-transformers-deep-dive/02-self-attention-from-scratch](../phases/07-transformers-deep-dive/02-self-attention-from-scratch/)
**My code:**
- [`labs/07-02-self-attention/my_attention.py`](../labs/07-02-self-attention/my_attention.py): attention, causal mask, multi-head attention. 6/6 checks.
- [`use_it.py`](../labs/07-02-self-attention/use_it.py): my weights loaded into `torch.nn.MultiheadAttention`. **MATCH to about 1e-15.**

**Run it:** `python labs/07-02-self-attention/my_attention.py`, then `python labs/07-02-self-attention/use_it.py`

---

## Remember these 3 things

1. **`Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`.**
   - Each token's query is compared with every key.
   - Softmax turns the scores into weights.
   - The output is the weighted blend of the values.
2. **Two guardrails:**
   - **Divide by √d_k**, or the scores grow with dimension, softmax goes one-hot, and gradients die (at d = 512: 0.91 average top weight unscaled vs 0.32 scaled).
   - The **causal mask** sets future scores to −∞, so token t sees only 0..t.
3. **Attention is order-blind and quadratic.**
   - Shuffle the tokens and you get the same outputs, shuffled. That's why transformers add **positional information** (learned positions or RoPE).
   - The n × n score matrix means double the prompt costs about **4×** for attention.

---

## In plain English

Every token starts as a vector that only knows about itself. "it" doesn't know whether it means the animal or the street. Attention lets each token **look at the whole sequence and pull in what's relevant**.

```text
A: "The animal didn't cross the street because it was too tired."  → "it" = animal
B: "The animal didn't cross the street because it was too wide."   → "it" = street
```

**No position-based rule can work.** "animal" and "street" sit at *identical positions* in both sentences. The only difference is "tired" vs "wide", which comes *after* "it". The right target depends on **content**, so the scores come from comparing content vectors.

**Why soft weights, not "pick one":** argmax has zero or undefined gradient, so the model couldn't *learn* where to look. A softmax-weighted blend changes smoothly, so backprop (03/03) can nudge it: "look a bit more at animal".

**Analogy: a soft database lookup.**

| Vector | Made by | Question | Database |
|--------|---------|----------|----------|
| **Query** q = x·Wq | learned matrix | "What am I looking for?" | the search |
| **Key** k = x·Wk | learned matrix | "What do I contain, for matching?" | the index |
| **Value** v = x·Wv | learned matrix | "What do I hand over if picked?" | the row content |

A database returns *one* exact match. Attention returns a **weighted blend of all rows**.

---

## Worked example (2-dimensional, "it" in sentence A)

```text
q_it = [1, 0]      k_animal = [2, 0]    k_street = [0, 2]    k_it = [0.5, 0.5]
```

| Step | animal | street | it |
|------|--------|--------|----|
| q · k | 2 | 0 | 0.5 |
| ÷ √2 | 1.41 | 0 | 0.35 |
| softmax | **0.629** | 0.153 | 0.218 |

"it" takes 63% of its new meaning from "animal". The output is `0.629·v_animal + 0.153·v_street + 0.218·v_it`.

---

## Why divide by √d_k

- Each q and k component is about unit variance, so the dot product has variance **d_k** and a typical size of **√d_k**. Dividing brings it back to about 1.
- Without it: [2, 0, 0.5] becomes something like [20, 0, 5] in high dimensions, and softmax gives ≈ [0.9999997, 2e-9, 3e-7], **essentially one-hot**.
- The softmax gradient involves p(1 − p) ≈ 0, which is **saturation**, the same failure as sigmoid in 03/03. The attention pattern stops learning.
- **Measured** (d_k = 512, 8 random tokens): average top weight **0.32 scaled vs 0.91 unscaled**.

---

## Causal mask

```text
token i may look at tokens 0..i       mask = np.tril(np.ones((n, n), dtype=bool))
     t0 t1 t2 t3
t0 [ ✓  ·  ·  · ]
t1 [ ✓  ✓  ·  · ]
t2 [ ✓  ✓  ✓  · ]
t3 [ ✓  ✓  ✓  ✓ ]
```

- Blocked scores become **−∞**, and `exp(−∞) = 0`, so those weights are **exactly 0**.
- **The guarantee my check proved:** changing token 3 by +100 left tokens 0–2's outputs **unchanged**. That's what makes left-to-right generation, and the KV cache (next lesson), possible.
- **Danger:** a row where *everything* is masked gives softmax 0/0 = **NaN**, which then spreads through the model. It happens with padding masks or a flipped mask.

---

## Multi-head attention

1. Project: Q = X·Wq, K = X·Wk, V = X·Wv, each (n, d_model).
2. Split the **columns** into h heads of d_head = d_model / h. **512 / 8 = 64.**
3. Run attention per head. Each head learns a different relationship (syntax, coreference, position, …).
4. Concatenate the heads, which gives (n, d_model), then multiply by Wo.

The total compute is about the same as a single full-width head, but you get h different attention patterns.

---

## My code

```python
def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)            # (n, n): row i = token i's scores for every token
    if mask is not None:
        scores[~mask] = -np.inf                # blocked → weight exactly 0
    weights = softmax(scores)                  # each row sums to 1
    return weights @ V, weights

def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))   # review fix: explicit (n, n)

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    d_model = X.shape[1]
    if d_model % n_heads:
        raise ValueError("d_model must divide evenly by n_heads")
    d_head = d_model // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    outs, ws = [], []
    for h in range(n_heads):
        s = slice(h * d_head, (h + 1) * d_head)       # column block for this head
        o, w = attention(Q[:, s], K[:, s], V[:, s], mask)
        outs.append(o); ws.append(w)
    return np.hstack(outs) @ Wo, ws
```

**Review points:**
1. I first wrote `np.ones(n, dtype=bool)`, which is **1-D**. It only worked because `np.tril` happens to broadcast it to (n, n). Write the shape explicitly.
2. A fully masked row gives NaN. Production code guards it.
3. Production reshapes to `(n, n_heads, d_head)` and does one batched matmul instead of a Python loop over heads.

---

## Attention is order-blind (my wrong prediction)

I predicted that shuffling the tokens would *change* the outputs. **Wrong.** The outputs come back as **the same rows, shuffled**.

- Each output is a weighted **sum** over all keys and values, and a sum doesn't care about order.
- So "dog" gets the identical vector in "dog bites man" and "man bites dog".
- **The fix:** add position information to the embeddings, either learned position vectors or **RoPE** (rotary, used by Llama and most current LLMs).

---

## Use it: PyTorch conventions that cause real bugs

I copied my weights into `torch.nn.MultiheadAttention` (bias off). Both the unmasked and causal runs **MATCH to about 1e-15**.

| Convention | Mine | PyTorch | If you get it wrong |
|------------|------|---------|---------------------|
| Weight layout | `X @ W` | stores `(out, in)` and computes `x @ W.T`, so load `W.T` | silently wrong outputs (error 3.9 in my run) |
| Q, K, V weights | three matrices | one `in_proj_weight` = `vstack([Wq.T, Wk.T, Wv.T])` | |
| Mask polarity | **True = allowed** | `attn_mask` **True = blocked** | tokens see only the **future**, and the last token's row is fully blocked, so **NaN** |

---

## Where this shows up in AI engineering

- **Cost of long prompts:** attention is O(n²) in prompt length, so long-context requests cost more than linearly in compute. That's the reason for FlashAttention, KV caching and prompt caching (next lesson).
- **Porting or loading weights** between frameworks: check the transpose and the mask polarity.
- **Debugging NaN in training or inference:** look for fully masked rows (padding) and flipped masks.
- **Interview:** "Explain attention", "Why √d_k?", "Why does a transformer need positional encoding?", "Why is it quadratic?". All four answers are in this sheet.

## Mistakes I made

- Thought a position rule could disambiguate "it" (animal and street are at the *same* positions).
- Explained soft weights as "rules fail". The real reason is that **argmax has no gradient**.
- Predicted that shuffling changes the outputs (they're permuted, not changed).
- Said an unflipped mask "masks all tokens". Actually it allows **only the future**, and blocks the last row completely, giving NaN.

## Words

- **Self-attention:** Q, K and V all come from the same sequence.
- **Attention weights:** a row-wise softmax over scaled scores. Each row sums to 1.
- **Causal (autoregressive) mask:** blocks future positions.
- **Head:** one attention computation on a d_head-wide slice.
- **Permutation-equivariant:** shuffle the inputs and the outputs are shuffled the same way.
- **RoPE:** rotary position embedding. Rotates q and k by an angle that depends on position.

## Connects to

- **Before:** [01/06](01-06-probability-and-distributions.md) (softmax, why soft beats argmax), [03/03](03-03-backpropagation.md) (saturation kills gradients), [01/14](01-14-norms-and-distances.md) (dot-product similarity), [05/22](05-22-embedding-models.md) (MaxSim is a cousin of the q·k comparison).
- **Next:** 07/07 GPT Causal Language Modeling. Stacking attention, MLP layers and residuals into a model that predicts the next token.

---

## Self-test

1. Write the attention formula. What shapes are Q·Kᵀ and the output for n tokens?
2. Why can't any position-based rule resolve "it" in the tired/wide sentences?
3. Why soft weights instead of picking the single best match?
4. q = [1, 0], keys [2, 0], [0, 2], [0.5, 0.5]: which token wins, with roughly what weight?
5. Why divide by √d_k? What breaks without it?
6. What does the causal mask do, and what did editing token 3 prove?
7. 8 heads, d_model = 512: what's d_head?
8. You shuffle the input tokens (no mask). What happens to the outputs, and what does that imply?
9. Loading weights into PyTorch: what two convention traps did you hit?

<details>
<summary>Answers</summary>

1. `softmax(Q·Kᵀ / √d_k)·V`. Q·Kᵀ is (n, n). The output is (n, d_v).
2. "animal" and "street" are at identical positions. Only content after "it" differs, so any position rule is wrong for one sentence.
3. Argmax has zero or undefined gradient, so "where to look" couldn't be learned. Softmax weights change smoothly.
4. "animal", at about 0.63 (0.629 / 0.153 / 0.218).
5. Dot products grow like √d_k, so softmax goes one-hot, p(1 − p) ≈ 0, and the gradients vanish (saturation).
6. It sets future scores to −∞, giving weight 0. Editing token 3 left tokens 0–2 unchanged, so no information flows from the future.
7. 64.
8. The same rows come back, shuffled (permutation-equivariant). Attention alone is order-blind, so positional information must be added.
9. PyTorch computes `x @ W.T` (store W.T, stacked into `in_proj_weight`), and its `attn_mask` uses True = blocked (flip the mask).

</details>
