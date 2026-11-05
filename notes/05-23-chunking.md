# 05/23 Chunking Strategies for RAG

> The chunk is both what gets *found* and what the LLM *reads*. If the cut is wrong, no embedding model or reranker can find the answer.

**Done:** 2026-10-25 · Quiz 3/3 · Bridge lesson 11 of 14
**Lesson:** [phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag](../phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/)
**My code:**
- [`labs/05-23-chunking/my_chunking.py`](../labs/05-23-chunking/my_chunking.py): fixed, recursive and parent–child chunkers plus parent dedupe. 7/7 checks.
- [`use_it.py`](../labs/05-23-chunking/use_it.py): my chunkers plus BGE retrieval on a 13-section contract, scored on hits *and* prompt tokens.

**Run it:**
- `python3 labs/05-23-chunking/my_chunking.py` (stdlib only)
- `python labs/05-23-chunking/use_it.py` (needs sentence-transformers and tiktoken)

---

## Remember these 3 things

1. **Too big, too small, wrong cut. Each fails differently.**
   - **Too big:** the model reads only its first ~512 tokens, and the rest gets diluted.
   - **Too small:** "Such notice…" has no context.
   - **Wrong cut:** "terminate this Agree" | "ment…".
2. **Recursive chunking is the default.** Split on paragraphs, then lines, then sentences, then spaces. It cuts at natural boundaries, so overlap rarely adds anything. Start at about 256 to 512 tokens for factoid queries, then measure.
3. **Score retrieval *and* prompt size.** A "hit" metric rewards stuffing: sending the whole document scored a perfect 1.0. Recursive 200 got the same hit@3 at **84 tokens instead of 370**.

---

## In plain English

The RAG ingestion pipeline: **documents → chunks → embed → index**. At query time, retrieve the top-k chunks and paste them into the prompt.

The chunk does two jobs, and they pull in opposite directions:
- **To be found**, it should be *focused*: one topic, so its vector isn't diluted.
- **To be useful**, it should be *complete*: enough context for the LLM to answer.

**Analogy:** cutting a book into index cards. One card per book and you can't find anything. One card per sentence and every card is a fragment. Cut at chapter and paragraph breaks and each card is findable *and* readable.

---

## The three failures (contract, section 12 on page 23)

```text
12. Termination.
12.1 Either party may terminate this Agreement with 30 days' written notice.
12.2 Such notice must be sent by certified mail to the address in Section 2.
```

Query: **"How can we cancel the contract?"**

| Mistake | What happens |
|---------|-------------|
| **Whole document as one chunk** | (1) **Input truncation:** the embedding model reads only the first ~512 tokens, so section 12 is *never read*. There's no error, and the vector describes the cover page. (2) Even a long-context model averages 3 sentences into thousands, which dilutes the signal. |
| **One sentence per chunk** | "Such notice must be sent…" is retrieved but useless: *such* notice? There's no subject, and nothing about termination. |
| **Fixed cut mid-word** | "…terminate this Agree" \| "ment with 30 days'…". Each half loses half the meaning, and the cut leaves junk tokens. |

**Don't confuse the two truncations:**
- **Matryoshka (05/22)** cuts the *output vector*. That's your choice.
- **Input truncation** cuts the *text* before the model reads it. That's silent, and it's dangerous.

---

## Six strategies

| Strategy | How it cuts | Fixes | Watch out |
|----------|------------|-------|-----------|
| **Fixed** | every N characters or tokens | the baseline | mid-word and mid-sentence cuts |
| **Recursive** (default) | `\n\n`, then `\n`, then `". "`, then `" "`, falling back only for oversized pieces | natural boundaries | size still needs tuning |
| **Sentence** | sentence boundaries, N per chunk | clean cuts | fragments with no context |
| **Semantic** | embed each sentence, cut where neighbour similarity drops | topic coherence | **needs a min-size floor**, or you get 40-token fragments. Slow. |
| **Parent–child** | embed small children, **return their parent** | precise retrieval plus full context | dedupe parents. Parents must be small relative to the doc. |
| **Contextual retrieval** (Anthropic) | an LLM writes a 50–100 word "where this sits" prefix, embedded *with* the chunk | missing context, **at retrieval time** | one LLM call per chunk at index time |
| **Late chunking** | embed the whole document at token level, *then* average the tokens inside each chunk | cross-chunk context, no LLM calls | needs a long-context embedder |

**Which part each fix improves:**
- **Parent–child fixes what the LLM *reads*.**
- **Contextual retrieval fixes what gets *found*.** The prefix ("…from Section 12, Termination…") is embedded with the chunk, so "Such notice…" now matches "cancel the contract".

**Chunk size by query type** (NVIDIA 2026):

| Query type | Chunk size |
|-----------|-----------|
| Factoid | 256 to 512 tokens |
| Analytical or multi-hop | 512 to 1024 tokens, or parent–child |
| Whole-section comprehension | 1024 to 2048 tokens |

**Overlap:**
- **Helps fixed chunking:** it rescued the mid-word cuts in my lab.
- **Adds little to recursive:** the cuts are already at boundaries.
- **Always costs:**
  - more chunks, so a bigger index and more embedding calls
  - **duplicate text in the top-k**, so you pay for the same sentences twice in the prompt
- **Doesn't fix cross-references:** "Section 2" stays elsewhere.

---

## My code

```python
def chunk_recursive(text, size, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep in text:
            pieces = text.split(sep)
            pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]]   # KEEP the separator, so nothing is lost
            chunks, buffer = [], ""
            for piece in pieces:
                if len(buffer) + len(piece) <= size:
                    buffer += piece                                    # pack greedily
                else:
                    if buffer:
                        chunks.append(buffer)
                    if len(piece) > size:                              # oversized: recurse with the REMAINING seps
                        chunks.extend(chunk_recursive(piece, size, seps[i + 1:]))
                        buffer = ""
                    else:
                        buffer = piece
            if buffer:
                chunks.append(buffer)
            return chunks
    return chunk_fixed(text, size)                                     # no separator at all
```

**The invariant to test:** `"".join(chunks) == text` and `max(len) <= size`. Dropping the separator fails the first. Recursing with *all* the separators again loops forever on the same one.

```python
def retrieve_parents(mapping, child_scores, top_k):
    top = sorted(zip(child_scores, mapping), key=lambda x: -x[0])[:top_k]
    seen, out = set(), []
    for _, e in top:
        if e["parent_idx"] not in seen:          # review fix: dedupe by INDEX, not text
            seen.add(e["parent_idx"])
            out.append(e["parent"])
    return out
```

**Review points:**
1. **Dedupe by `parent_idx`, not by parent text.** Repeated boilerplate parents ("Intentionally left blank") would otherwise be merged and silently dropped.
2. Use `enumerate(seps)` and `seps[i + 1:]` rather than `seps.index(sep)`.
3. Give `ValueError` a message.

---

## My lab results (13-section contract, 370 tokens, 10 queries, BGE-small)

| Strategy | Chunks | hit@1 | hit@3 | Prompt tokens (top-3) |
|----------|--------|-------|-------|-----------------------|
| Whole document | 1 | 1.0 | 1.0 | 370 |
| Fixed 200 | 9 | 0.7 | **0.7** | 131 |
| Fixed 200, overlap 50 | 11 | 0.7 | 1.0 | 133 |
| **Recursive 200** | 14 | 0.8 | **1.0** | **84** |
| Parent 800 / child 150 | 15 | 0.8 | 1.0 | 370 |

**Reading it:**
- **Fixed 200** missed "cancel the contract", "cap on damages" and the multi-hop query, because the mid-word cuts broke those clauses.
- **Overlap** rescued hit@3 for +2 tokens but didn't improve hit@1.
- **Recursive 200** is the real winner: perfect hit@3, multi-hop solved (§12.2 and §2 both in the top 3), **4.4× cheaper** than sending everything.
- **Whole document's "perfect" score is fake.**
  - The hit metric only asks whether the answer is *somewhere* in the prompt, so stuffing always wins.
  - A real 50-page contract is about 15,000 tokens per query, BGE-small would only read the first 512, and extra context makes the LLM worse ("context cliff" around 2.5k tokens).
- **Parent 800 degenerated into the whole document.** The contract has only about 3 parents of 800 characters, and the top-3 deduplicated parents *are* the document. **Parents must be small relative to the document.**

---

## Pitfalls

- Evaluating only on factoid queries. Multi-hop queries pick different winners, so stratify the eval set.
- Semantic chunking with no minimum size.
- No min/max size enforcement at all.
- Overlap by habit: measure it.
- **Chunks that span two documents.** Always chunk per document.
- A hit metric without a cost or size measure.

## Where this shows up in AI engineering

- **Every RAG ingestion pipeline:** LangChain's `RecursiveCharacterTextSplitter` is the version I just built.
- **Interview:** "Our RAG returns the cover page for contract questions." Check for input truncation and whole-document chunks. Switch to recursive at 256–512 tokens, consider parent–child or contextual retrieval for cross-references, and evaluate hit@k *and* prompt tokens.
- **Cost:** chunk size × k = tokens per query × queries per day = the bill.

## Words

- **Chunk:** the unit that is embedded, indexed, retrieved and shown to the LLM.
- **Input truncation:** text past the model's maximum length is silently dropped.
- **Context cliff:** answer quality drops as retrieved context grows (around 2.5k tokens in 2026 studies).
- **Multi-hop query:** needs facts from two or more places (§12.2 *and* §2).

## Connects to

- **Before:** [05/22 Embedding Models](05-22-embedding-models.md) (what embeds the chunks, MRR and recall@k) and [05/19 Tokenization](05-19-subword-tokenization.md) (chunk sizes and costs are in tokens).
- **Next:** 07/02 Self-Attention from Scratch. What the model actually does with the chunks you send it.

---

## Self-test

1. Name the three ways chunking fails on the contract, with one consequence each.
2. Matryoshka truncation vs input truncation: what gets cut in each?
3. Write the separator order recursive chunking tries, and the invariant your tests checked.
4. Why does overlap help fixed chunking but add little to recursive? What does it cost?
5. Parent–child vs contextual retrieval: which fixes *finding* and which fixes *reading*?
6. In my lab, why did "whole document" score a perfect 1.0, and why is that misleading?
7. Why did parent 800 cost the same tokens as the whole document?

<details>
<summary>Answers</summary>

1. Too big: input truncation (section 12 never read) plus dilution. Too small: "Such notice…" has no context. Mid-word cut: each half loses meaning.
2. Matryoshka cuts the output *vector* (your choice). Input truncation cuts the *text* before the model reads it (silent).
3. `\n\n`, then `\n`, then `". "`, then `" "`, then fixed. `"".join(chunks) == text` and every chunk ≤ size.
4. Recursive already cuts at boundaries, so there's no broken sentence to rescue. Overlap costs more chunks (index size, embedding calls) and duplicate text in the top-k.
5. Contextual retrieval fixes finding (its prefix is embedded with the chunk). Parent–child fixes reading (you return the bigger parent).
6. The hit metric only checks whether the answer is somewhere in the prompt, so stuffing always wins. It ignores cost, the 512-token input truncation, and context noise.
7. There are only about 3 parents of 800 characters in a 1,700-character contract, and the top-3 deduplicated parents are the whole document.

</details>
