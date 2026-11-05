# 07/12 KV Cache and Flash Attention

> Training is parallel and limited by compute. Generation is serial and limited by memory. The KV cache stops recomputing the past, Flash Attention stops shuffling the N×N score matrix through slow memory, and prefix caching shares the past between requests.

**Done:** 2026-10-28 · Quiz 3/3 · Bridge lesson 14 of 14 (bridge complete)
**Lesson:** [phases/07-transformers-deep-dive/12-kv-cache-flash-attention](../phases/07-transformers-deep-dive/12-kv-cache-flash-attention/)
**My code:**
- [`labs/07-12-kv-cache/my_kv.py`](../labs/07-12-kv-cache/my_kv.py): `kv_cache_bytes`, `KVCache.copy`, `prefill`, `decode`, `tiled_attention` (online softmax). 9/9 checks on the first run.
- [`use_it.py`](../labs/07-12-kv-cache/use_it.py): my tiled attention vs `F.scaled_dot_product_attention`, greedy GPT-2 with and without `past_key_values`, and the real cache size vs my formula. **All three match.**

**Run it:** `python labs/07-12-kv-cache/my_kv.py`, then `python labs/07-12-kv-cache/use_it.py`

---

## Remember these 3 things

1. **Cache K and V, never Q.**
   - The causal mask means adding a token never changes anything computed for earlier positions, so their K and V are fixed forever.
   - The new token only needs its own `q`, plus K and V for everything so far. Past queries are never used again.
   - **Prefill** processes the whole prompt in one parallel pass (limited by compute; this is time-to-first-token). **Decode** handles one token per step and reads all the weights plus the whole cache each step (limited by memory bandwidth).
2. **KV cache memory, not the weights, decides how many users fit on a GPU.**
   - `bytes = 2 × layers × kv_heads × d_head × bytes_per_value × tokens × batch`
   - Llama-3-8B in bf16: **128 KiB per token**, **1 GiB per 8K user**, so about 60 users fit in the 64 GB left on an 80 GB card. At 32K context that's about 15.
   - GQA (8 KV heads instead of 32) cuts the cache 4×. An fp8 KV cache halves it again.
3. **Flash Attention is exact, and the speed comes from less memory traffic.**
   - It tiles Q/K/V through on-chip SRAM, keeping a running max `m`, sum `s` and accumulator `acc`, and never writes the N×N matrix to HBM.
   - When a new block raises the max, it multiplies the old `s` and `acc` by `exp(m_old − m_new)`. That's exact, so my version matched torch to **1.1e-16**.
   - The FLOPs are the same. The cut is in HBM round trips (SRAM is about 30 TB/s, HBM about 3 TB/s).

---

## In plain English

**The KV cache** is like taking notes in a meeting. Without notes, every time someone new speaks you'd replay the entire recording from the start to remember what everyone said. With notes, you just add one line per new speaker and glance at the notes. The K vectors are the "what was this about" index on each note, and the V vectors are the note contents. The new speaker's question (Q) is only needed once, right now.

**Prefix caching** is photocopying a shared stack of notes. Every request starts with the same 6,000-token system prompt, so you take those notes once, and each new request gets a **copy** to add its own lines to. If you hand everyone the *same* stack instead of a copy, user 3 reads user 1's conversation. That's a wrong answer and a privacy leak.

**Flash Attention** is adding up a long receipt in chunks that fit on a sticky note, instead of copying the whole receipt to a whiteboard across the room, then reading it back, then writing the result. The sticky note is SRAM and the whiteboard is HBM. The answer is the same, but there's far less walking.

---

## Worked example 1: cache size (computed in the lesson)

```text
Llama-3-8B: 32 layers, 8 KV heads (GQA), d_head 128, bf16 = 2 bytes
per token  = 2 × 32 × 8 × 128 × 2          = 131,072 bytes = 128 KiB
8K user    = 131,072 × 8,192               = 1,073,741,824 bytes = 1 GiB
users/GPU  = (80 GB − 16 GB weights) / 1.07 GB ≈ 60 (fewer in practice: activations, 0.9 utilisation)

GPT-2 (my Use It): 12 layers, 12 KV heads (no GQA), d_head 64, fp32 = 4 bytes
157 tokens = 2 × 12 × 12 × 64 × 4 × 157    = 11,575,296 bytes   ← matched the real DynamicCache exactly
```

## Worked example 2: work saved by the cache (my toy, measured)

```text
prompt P = 5, new tokens N = 20
naive projections  = 5 + 6 + ... + 24        = 290     (whole prefix every step)
cached projections = 5 (prefill) + 19        = 24      (last token never needs projecting)

prefix caching: 30-token system prompt, 3 users × 4 tokens, 6 new each
naive             = 657 projections
prefix-cached     = 30 + 3 × (4 + 6 − 1)     = 57
```

## Worked example 3: real GPT-2, 150 greedy tokens on CPU (measured)

```text
no cache 7.2 s   cached 1.6 s   speedup 4.4×     (same tokens either way)
projection ratio would be 12,375 / 157 ≈ 79×
```

Why only 4.4×: every forward pass reads all 124M weights (about 500 MB) whatever the number of tokens, and the naive pass's extra rows run *in parallel* inside one matmul. The cache's advantage grows with length, since naive is O(N²).

---

## Formulas / cheat sheet

| Thing | Formula | Symbols |
|-------|---------|---------|
| KV cache bytes | `2 · L · h_kv · d_head · b · T · B` | 2 = K and V, L = layers, h_kv = KV heads, b = bytes per value (2 bf16, 1 fp8), T = tokens, B = batch |
| Decode step | `out = softmax(q_t Kᵀ / √d) V` | q_t = new token's query only, K and V = the whole cache |
| Online softmax, per block | `new_m = max(m, max(scores))`; `r = exp(m − new_m)`; `s = s·r + Σ exp(scores − new_m)`; `acc = acc·r + exp(scores − new_m) @ V_blk` | final output = `acc / s` |
| Naive vs cached work | `Σ_{i=0}^{N−1}(P + i)` vs `P + N − 1` | P = prompt length, N = new tokens |

---

## My code (with line notes)

```python
def prefill(cache, X):
    Q, K, V = project(X)          # ONE parallel pass over the whole prompt
    for k, v in zip(K, V):
        cache.append(k, v)        # store every prompt token's K and V
    return Q[-1]                  # only the last query is needed to start decoding

def decode(cache, q, n_new):
    outs = np.zeros((n_new, D))
    for _ in range(n_new):                         # fix: name it `step`, since it's used as an index
        out = attend(q, cache.K, cache.V)          # new query vs ALL cached keys/values
        outs[_] = out
        if _ < n_new - 1:                          # the last token is never fed back, so skip it
            Q, K, V = project(next_input(out).reshape(1, -1))   # project ONLY the new token
            cache.append(K[0], V[0])
            q = Q[0]
    return outs

def tiled_attention(q, K, V, tile=4):
    m, s, acc = -np.inf, 0.0, np.zeros(D)          # fix: np.zeros(V.shape[1]); d_v can differ from d_k
    for start in range(0, K.shape[0], tile):
        scores = K[start:start+tile] @ q / np.sqrt(D)
        new_m = max(m, scores.max())
        scale = np.exp(m - new_m)                  # exp(-inf) = 0 makes the first block work
        s   = s * scale + np.sum(np.exp(scores - new_m))
        acc = acc * scale + np.exp(scores - new_m) @ V[start:start+tile]
        m = new_m
    return acc / s
```

Hugging Face version (`use_it.py`): prefill with `out = model(ids, use_cache=True)`, then feed **only** `next_id` of shape (1, 1) with `past_key_values=cache`.

**Fixes still pending:**
- `attend(q, cache.K, cache.V)` passes lists, so NumPy copies the whole cache into an array every step. That's a hidden O(t) cost. Real engines preallocate tensors and write into a slot (PagedAttention uses 16-token blocks).
- In `use_it.py`, numpy, torch and F are imported a second time inside the TODO block.

---

## Where this shows up in AI engineering

- **Prompt caching in the Claude and OpenAI APIs** is a saved KV cache for the prefix. Cached input tokens are billed at a fraction of the normal price and return faster. To get hits, **put the stable content first** (system prompt, tool definitions, documents) and anything that changes last. A timestamp at the top breaks every hit.
- **Capacity planning:** concurrent users ≈ (GPU memory − weights) / (KV bytes per token × context). Doubling the context limit halves the number of users.
- **vLLM flags:** `--enable-prefix-caching`, `--kv-cache-dtype fp8`, `--max-model-len` (caps per-user cache), `gpu_memory_utilization`.
- **Latency metrics:** time-to-first-token = prefill, tokens/sec = decode. They have different bottlenecks, so they need different fixes.
- **Related tricks:** continuous batching (swap finished requests out mid-batch), PagedAttention (cache in blocks, so no fragmentation and shared prefixes), speculative decoding (a draft model proposes k tokens and the big model verifies them in one pass; this speeds up *decode*, not prefill).

---

## Mistakes to avoid (ones I made or nearly made)

- **Thinking a new last token changes every logit row.** With the causal mask, only the last row changes. That's exactly why the cache is valid. (Warm-up miss, 2026-10-28.)
- **Re-feeding the full ids together with `past_key_values`.** The model *appends* them as new positions: a 5-token prompt became an 11-token cache, the prompt was duplicated with shifted positions, and the next token changed from `' contents'` to `' K'`. No error. I predicted 5. Once you have a cache, **feed only the new tokens.**
- **Sharing one cache object between requests instead of copying it.** Request 1 is fine. Requests 2 and 3 attend to earlier users' messages: wrong outputs and a cross-tenant leak.
- **Expecting wall-clock speedup to equal the work ratio.** 79× fewer projections gave 4.4× on GPT-2 on CPU, because the per-step weight reads are a fixed cost.
- **Calling Flash Attention an approximation, or saying it does fewer FLOPs.** It's exact, does the same FLOPs and makes fewer HBM trips.

---

## Words

- **KV cache:** stored K and V vectors for every token seen so far, per layer and KV head.
- **Prefill / decode:** a parallel pass over the prompt / one-token-at-a-time generation.
- **GQA (grouped-query attention):** many query heads share a few K/V heads, which shrinks the cache.
- **HBM / SRAM:** large, slow GPU memory (80 GB, ~3 TB/s) / small, fast on-chip memory (~256 KB per SM, ~30 TB/s).
- **Online (running) softmax:** softmax computed block by block with a running max and sum, rescaled exactly.
- **Prefix caching:** reusing the KV cache of an identical prompt prefix across requests.
- **PagedAttention:** KV cache in fixed 16-token blocks with a page table (vLLM).
- **Continuous batching:** new requests join the batch as soon as old ones finish.

## Connects to

- [07/02 Self-Attention](07-02-self-attention.md): `softmax(QKᵀ/√d)V`, the causal mask, O(n²).
- [07/07 GPT](07-07-gpt.md): serial generation is the "autoregressive tax" the cache reduces.
- [01/06 Probability](01-06-probability-and-distributions.md): stable softmax subtracts the max; online softmax does the same thing one block at a time.
- [05/19 Tokenization](05-19-subword-tokenization.md): cache cost is per *token*, so count tokens.

---

## Self-test

1. **Why cache K and V but not Q?**
   <details><summary>Answer</summary>The new token attends with its own query against every earlier key and value. Earlier queries were only used to make earlier outputs, which are already done and never change under the causal mask.</details>
2. **Llama-3-8B, bf16, 32 layers, 8 KV heads, d_head 128: how much KV cache for 4 users at 16K context?**
   <details><summary>Answer</summary>128 KiB per token × 16,384 = 2 GiB per user, so 8 GiB for 4 users.</details>
3. **You raise the context limit from 8K to 32K on the same GPU. What happens to the number of concurrent users?**
   <details><summary>Answer</summary>It drops to about a quarter (roughly 60 to 15 for Llama-3-8B on 80 GB), because cache memory grows linearly with tokens.</details>
4. **A teammate calls `model(full_ids, past_key_values=cache)` with a 5-token prompt plus 1 new token. What's the cache length, and what goes wrong?**
   <details><summary>Answer</summary>11 = 5 + 6. The prompt is stored twice with wrong positions, so the next token is silently wrong. Feed only the new token.</details>
5. **Is Flash Attention exact? Where does its speedup come from?**
   <details><summary>Answer</summary>Yes, up to float rounding order, because the running-max rescale by exp(m_old − m_new) is exact. The speedup comes from keeping tiles in SRAM and never writing the N×N scores to HBM. The FLOPs are the same.</details>
6. **Your agent's system prompt starts with "Current time: 14:03:22". Why is your prompt-cache hit rate zero, and what's the fix?**
   <details><summary>Answer</summary>The cache only hits on an identical token prefix. The timestamp changes every request, so it changes every K and V vector after it. Move anything that changes to the end, after the stable system prompt and tools.</details>
7. **The cache cut projections 79× but GPT-2 only got 4.4× faster. Why?**
   <details><summary>Answer</summary>Every forward pass reads all the weights no matter how many tokens it processes, which is a fixed cost per step. The naive version's extra tokens also run in parallel inside one matmul, so they cost much less than their count suggests. The gap grows with length.</details>
