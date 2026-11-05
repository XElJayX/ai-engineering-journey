# 11/05 Context Engineering: windows, budgets, ordering, history

> Everything the model reads, plus its answer, must fit in one window. The assembler fits the fixed parts first (system, query, answer reserve), then fills the rest with the elastic parts by priority (docs by score, history newest-first). Then **verify the server actually saw it**: Ollama silently cut my 4,645-token prompt to 2,048.

**Done:** 2026-11-03 · Quiz 3/3 · Route row 5 of 109
**Lesson:** [phases/11-llm-engineering/05-context-engineering](../phases/11-llm-engineering/05-context-engineering/)
**My code:**
- [`labs/11-05-context-engineering/my_context.py`](../labs/11-05-context-engineering/my_context.py): `reorder_edges`, `select_docs` (skip-and-continue), `window_history` (stop at first misfit), `assemble` (fixed → docs → history, plus my own re-measure-and-trim safety loops). 17/17 checks.
- [`experiment.py`](../labs/11-05-context-engineering/experiment.py): llama3.2 3B on 24 fictional filings. A: truncation; B: lost in the middle; C: top-3 vs 8 vs 24.

**Run it:** `python labs/11-05-context-engineering/my_context.py`, then `python labs/11-05-context-engineering/experiment.py` (needs `ollama serve`; ~3 min)

---

## Remember these 3 things

1. **Input + output share one window. Fixed parts first, elastic parts by priority.**
   - Fixed: system prompt, current query, **generation reserve**, needed tool schemas. If they don't fit, raise an error. Never cut them.
   - Elastic: retrieved docs (best score first, whole chunks, skip what doesn't fit), history (newest first, whole turns, **stop** at the first misfit, summarize what's dropped).
   - Layout: system first, history, docs, **question last**. Docs best-at-the-edges (`[r0, r2, r4, r3, r1]`).
2. **The server's window can silently differ from yours. Assert on it.**
   - Ollama's default `num_ctx` = **2,048**. My 4,645-token prompt became **2,050**, cut from the **front**: the system prompt (rule broken) and my best doc (placed first by reordering) were deleted. No error.
   - Check on every call: `if seen < 0.95 * sent: alert`, where seen = `prompt_eval_count` / `usage.prompt_tokens` / `usage.input_tokens`. One-sided, with tolerance for the tokenizer mismatch (4,645 vs 4,662 is normal).
3. **More context costs linearly and often buys nothing. Distractors beat position.**
   - top-3 → top-24: accuracy 8/8 → 8/8, but **7× the tokens, 6.9× the latency**.
   - Lost-in-the-middle **didn't show** on llama3.2 (8/8 at every position, 4.6K tokens, exact-name lookup). On a hard set (48 near-duplicate names, paraphrased metrics) it collapsed **everywhere** (0/6 at position 1). Measure on your own model and task; keep reordering since it's free, but cutting distractors is the bigger win.

---

## In plain English

The context window is a **suitcase with a fixed size**, and the model's answer has to fit in it too. Some things are non-negotiable: passport (system rules), ticket (the question) and room to bring something back (the answer reserve). Those go in first, and if they don't fit, you don't travel. Then you pack the optional things in order of importance: the best documents first, the most recent conversation next. Never pack half a shirt (half a table). If something big doesn't fit, a smaller useful item might. For conversation history, though, you keep a continuous stretch of the most recent turns, because a story with a missing middle chapter doesn't make sense.

Then the trap: the airline (the model server) might have a **smaller suitcase limit than you assumed**, and instead of rejecting your bag, it **silently throws out whatever's at the front**, which is exactly where you packed the most important things. So you check what actually arrived.

---

## Worked example

**Budget arithmetic** (8,192 window, system 600, query 80, reserve 1,000, chunks 400):
8,192 − (600 + 80 + 1,000) = **6,512** available → 6,512 / 400 = 16.28 → **16 chunks** (6,400 tokens, 112 spare). Floor it: never send a partial chunk.

**Edge reordering** (scores A 0.9, B 0.7, C 0.5, D 0.3, E 0.1):
ranked A B C D E → even ranks from the front: A, C, E; odd ranks reversed at the back: D, B → **[A, C, E, D, B]**. The strongest two sit at the two edges, the weakest in the middle.

**Token counting:** a 3-row SEC table is **37 tokens**; `words × 1.3` says 27. Numbers and pipes blow up word-based estimates.

**My experiment** (llama3.2 3B, T=0, 24 fictional filings = 4,352 tokens):

| Part | Setup | Result |
|---|---|---|
| A | All 24 docs, no `num_ctx` | model saw **2,050** of 4,645 tokens · wrong answer · `Source: EDGAR` rule dropped |
| A | Same, `num_ctx=8192` | saw **4,662** · correct `$728 million [1]` · rule followed |
| B | Answer doc at position 1/6/12/18/24 of 24 | **8/8 at every position** |
| B (hard pilot) | 48 docs, near-duplicate names, paraphrased metrics | 0/6 at position 1, 3/6 at 24, 2/6 at 48 |
| C | top-3 / top-8 / top-24 (answer = rank #2) | 8/8 each · 665 / 1,617 / 4,660 tokens · 2.0 / 3.9 / 13.7 s |

My predictions: B 8/4/8 (real 8/8/8), C 8× slower (real 6.9×), A hedged ("uses its default") with no number.

---

## Formulas / cheat sheet

- **available** = window − (system + query + reserve + per-message overhead).
- **doc budget** = min(available − wrapper tokens, doc_cap). Chunk cost = tokens(chunk) + tag overhead (~12).
- **history budget** = what's left after docs.
- **Token counts aren't additive:** count(a + b) ≠ count(a) + count(b) at the join. So re-measure the final messages and trim if over (history first, then the lowest-scored doc).
- **Truncation check:** `seen < 0.95 × sent` → alert. Log both on every call.
- **Window settings by stack:** Ollama `options.num_ctx`; vLLM `--max-model-len`; HF tokenizer `truncation=True, max_length=...` (silent too); hosted APIs return an error instead of truncating, but proxies and frameworks may trim for you.

---

## My code (what I wrote and why)

```python
for turn in reversed(turns):                       # newest first
    cost = count_tokens(turn["content"]) + MSG_OVERHEAD
    if cost + token_used > budget:
        break                                      # STOP: no holes in a conversation
    token_used += cost
    kept.append(turn)
kept = kept[::-1]                                  # back to chronological
dropped = turns[:len(turns) - len(kept)]           # always the older prefix
```

```python
total = sum(count_tokens(m["content"]) + MSG_OVERHEAD for m in messages)
while total > window - reserve and len(kept) > 0:   # my safety loop: re-measure the REAL total,
    dropped.append(kept.pop(0))                     # trim oldest history first...
    ...
while total > window - reserve and len(selected) > 0:
    selected.pop(); selected_scores.pop()           # ...then the lowest-scored doc
```

`select_docs` sorts by score (stable), adds `count_tokens(doc) + per_doc_overhead`, and skips what doesn't fit. `reorder_edges` puts even ranks at the front and odd ranks reversed at the back.

**Fixes still to make:**
- The report's units are inconsistent: `system`, `query` and `history` include `MSG_OVERHEAD`, `docs` doesn't, so the parts don't sum to `input_total`.
- `raise ContextOverflow()` has no message. Include `fixed` and `window`.

---

## Where this shows up in AI engineering

- **Project 1:** an assembler with a doc cap tuned on the golden set (top-3 to top-5 after reranking, not top-20), a generation reserve, and a sent-vs-seen check on every call, logged to the trace.
- **Project 2 (agent):** tool results and long logs blow the window fastest. Budget tool outputs, summarize the dropped history, keep the system rules first.
- **Serving:** fewer input tokens = less prefill and less KV cache (07/12), so more users per GPU and a lower bill (11/11 and 11/15 next).
- **Memory:** short-term = the window, long-term = facts loaded at session start (`CLAUDE.md`, `LEARNING.md`), episodic = embedded past interactions retrieved by similarity.

---

## Mistakes to avoid

- **Not setting the server's window.** Ollama defaulted to 2,048 and cut from the front: system prompt and best doc gone, no error.
- **Exact-equality checks on token counts.** Tokenizers and chat templates differ (4,645 vs 4,662). Use a one-sided check with tolerance.
- **Assuming lost-in-the-middle always shows up.** It didn't on llama3.2 for simple lookups. Measure; hard distractors were the real problem.
- **Hedging a prediction** ("it uses its default"). Commit to a number: here, "2,048 tokens, so the front gets cut".
- **History that skips ahead:** you keep a reply whose question was dropped, so the conversation has a hole.
- **Docs that stop at the first misfit:** one big chunk blocks a smaller one that holds the answer.

---

## Words

- **Context window:** the max tokens (input + output) in one call.
- **Generation reserve:** tokens held back for the answer.
- **Fixed vs elastic parts:** must-fit (system, query, reserve) vs trimmable (docs, history).
- **Lost in the middle:** lower accuracy for information in the middle of long contexts (Liu et al. 2023); model- and task-dependent.
- **Sliding window:** the last N turns verbatim, older ones dropped or summarized.
- **Tool pruning:** only include the tool schemas relevant to this query.
- **`num_ctx` / `max_model_len`:** the server-side window. If it's smaller than your budget, input gets silently truncated.

## Connects to

- [05-19 Subword Tokenization](05-19-subword-tokenization.md): count tokens, never estimate (the table is 37, not 27).
- [07-12 KV Cache](07-12-kv-cache.md): every input token costs prefill and cache memory.
- [11-04 Embeddings](11-04-embeddings.md): scores aren't calibrated, so watch for the score cliff, not a fixed threshold.
- [11-01 Prompt Engineering](11-01-prompt-engineering.md): rules in system, data in tagged user content.
- Next: 11/06 RAG (retrieval + this assembler + generation, end to end).

---

## Self-test

1. **Window 16,384, system 900, query 150, reserve 2,000, chunks 512 tokens + 12 tag tokens. Max chunks?**
   <details><summary>Answer</summary>16,384 − 3,050 = 13,334 available. Each chunk costs 524. 13,334 / 524 = 25.4, so 25 chunks (13,100 tokens).</details>
2. **Your server reports 2,050 prompt tokens; you counted 4,645. What happened and what got lost?**
   <details><summary>Answer</summary>The server window is ~2,048 (Ollama's default `num_ctx`), and it cut silently from the front. The front holds the system prompt (rules lost) and, with edge reordering, the best doc. Fix: set the window explicitly and alert when seen < 0.95 × sent.</details>
3. **Why do docs skip a misfit and keep going, but history stops at the first misfit?**
   <details><summary>Answer</summary>Docs are independent: a later, smaller chunk can still help. History is a sequence: skipping creates a hole (a reply with its question missing), which confuses the model.</details>
4. **Reorder scores [0.2, 0.9, 0.5, 0.7] for docs A, B, C, D with best-at-edges.**
   <details><summary>Answer</summary>Ranked: B (0.9), D (0.7), C (0.5), A (0.2). Even ranks at the front: B, C. Odd ranks reversed at the back: A, D. Result: [B, C, A, D].</details>
5. **top-24 vs top-3 gave the same 8/8. Why still send fewer, and when would more actively hurt?**
   <details><summary>Answer</summary>7× the tokens and 6.9× the latency for no gain, paid on every query. More hurts with hard distractors (FY2023 vs FY2024, near-duplicate company names): my hard pilot collapsed to 0-50%.</details>
6. **Why re-measure the final messages instead of trusting the sum of the parts?**
   <details><summary>Answer</summary>BPE counts aren't additive: tokens merge or split differently at the joins between pieces. Re-measure the rendered messages and trim (oldest history, then the lowest-scored doc) until it fits.</details>
7. **Lost-in-the-middle didn't show on llama3.2. Is reordering pointless?**
   <details><summary>Answer</summary>No. It's free and helps on models and tasks where the effect exists, and Liu et al. showed 10-20 points there. But measure: on my model, cutting distractors mattered more than position.</details>
