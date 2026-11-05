# Lesson 11/06 RAG: wire YOUR earlier pieces into one pipeline, then grade every answer into a failure bucket.
# Run: python labs/11-06-rag/my_rag.py      (self-checks use a fake embedder and a fake LLM; no model needed)
# Reuses: chunk_recursive (05/23), exact_topk + normalize (11/04), assemble + reorder_edges (11/05).
# The "Use it" run (experiment.py) indexes fictional 10-Ks with bge-small and answers with llama3.2.
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("05-23-chunking", "11-04-embeddings", "11-05-context-engineering"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from my_chunking import chunk_recursive          # noqa: E402  (your 05/23 code)
from my_vector_index import exact_topk, normalize  # noqa: E402  (your 11/04 code)
from my_context import assemble, reorder_edges     # noqa: E402  (your 11/05 code)

SYSTEM = ("You are a financial analyst assistant. Answer ONLY from the documents provided. "
          "Reply with the figure and the id of the doc you used, like: $1,234 million [3]. "
          "If the documents do not contain the answer, reply exactly: I don't have enough information.")
ABSTAIN = "i don't have enough information"


def is_abstain(reply):
    # Given. Models vary the apostrophe and casing; normalise both.
    return ABSTAIN in reply.lower().replace("’", "'")


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def build_index(docs, embed_docs, size=600, header=False):
    # TODO 1: the OFFLINE half. docs = [{"title": str, "text": str}, ...].
    #   - Split every doc's text with your chunk_recursive(text, size).
    #   - If header is True, prefix each chunk with "[<title>] " BEFORE embedding (and keep it in the stored text).
    #     (A 10-K paragraph often says "the Company" and never names it; the header puts the name back.)
    #   - Embed all chunk texts in ONE call: embed_docs(list_of_texts) -> (n, d) array; normalize the rows.
    # Return (chunks, D) where chunks = [{"doc": doc_index, "text": chunk_text}, ...] and D is (n, d) float32.
    chunks = []
    for i,doc in enumerate(docs):
        pieces = chunk_recursive(doc["text"],size)
        for piece in pieces:
            if header:
                piece = f"[{doc['title']}] " + piece
            chunks.append({"doc": i, "text": piece })
    texts = [ chunk["text"] for chunk in chunks]
    D = embed_docs(texts)
    D = np.asarray(D, dtype=np.float32)
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    D = D / np.maximum(norms, 1e-12)
    return (chunks,D)

def rag_answer(question, chunks, D, embed_query, llm, k=5, window=8192, reserve=512):
    # TODO 2: the ONLINE half.
    #   1. q = normalize(embed_query(question)) reshaped to (1, d); top-k with your exact_topk(q, D, k).
    #   2. messages, report = assemble(SYSTEM, question, [texts of the k chunks], [their scores], [], window, reserve)
    #   3. reply = llm(messages)
    #   4. SHOWN ORDER. Your assemble put the docs through reorder_edges, so the model's "[3]" means the 3rd doc
    #      it SAW, not the 3rd best retrieval hit. Compute shown_ids = the chunk ids in the order the model saw
    #      them (hint: call reorder_edges on the chunk ids with the same scores). Assert report["n_docs"] == k so
    #      you know assemble didn't drop any.
    # Return {"reply": reply, "retrieved": [chunk ids, best first], "shown": shown_ids, "input_tokens": report["input_total"]}
    q = np.asarray(embed_query(question), dtype = np.float32)
    q = q.reshape(1,-1)
    norms = np.linalg.norm(q,axis = 1, keepdims = True)
    q = q / np.maximum(norms, 1e-12)
    candidate_ids, candidate_scores = exact_topk(q,D,k)
    candidate_ids = np.asarray(candidate_ids).reshape(-1)
    candidate_scores = np.asarray(candidate_scores).reshape(-1)
    retrieved_texts = [
    chunks[idx]["text"]
    for idx in candidate_ids
    ]
    messages, report = assemble(SYSTEM, question, retrieved_texts, candidate_scores,[],window,reserve)
    assert report["n_docs"] == k
    reply = llm(messages)
    shown_ids = reorder_edges(candidate_ids,candidate_scores)
    return {
    "reply": reply,
    "retrieved": candidate_ids.tolist(),
    "shown": list(shown_ids),
    "input_tokens": report["input_total"]
}


def parse_citations(reply):
    # TODO 3: return the sorted, de-duplicated list of integer doc ids cited in the reply.
    #   "$728 million [3]" -> [3]     "see [2] and [5], also [2]" -> [2, 5]     "[1, 4]" -> [1, 4]     no brackets -> []
    #   Ignore brackets that don't hold only numbers, e.g. "[Kestrel Dynamics 10-K FY2024]" -> no id.
    matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", reply)
    ids = set()
    for match in matches:
        for num in match.split(","):
            ids.add(int(num.strip()))
    return sorted(ids)



def citation_ok(reply, value, shown_ids, chunks):
    # TODO 4: is the answer's citation checkable AND supporting? True only if ALL of these hold:
    #   - at least one id is cited
    #   - every cited id is in range 1..len(shown_ids)  (a model citing [7] when it saw 5 docs is making it up)
    #   - at least one cited doc's text contains the answer figure, formatted f"{value:,}"  (e.g. "1,284")
    # Map a cited id i to its chunk with shown_ids[i - 1].
    cited_ids = parse_citations(reply)
    if not cited_ids:
        return False
    formatted_value = f"{value:,}"
    for cited_id in cited_ids:
        if cited_id < 1 or cited_id > len(shown_ids):
            return False
    for cited_id in cited_ids:
        chunk_id = shown_ids[cited_id - 1]
        chunk_text = chunks[chunk_id]["text"]
        if formatted_value in chunk_text:
            return True

    return False



def grade(reply, value, gold_ids, retrieved_ids):
    # TODO 5: put one answer in exactly one bucket.
    #   value is None -> the question is UNANSWERABLE from the corpus:
    #       abstained -> "correct_abstain"      anything else -> "hallucination"
    #   value is a number -> answerable. correct = f"{value:,}" appears in the reply AS A WHOLE NUMBER (not inside
    #   a bigger one) AND it did not abstain.
    #   gold_retrieved = any gold chunk id is in retrieved_ids.
    #       gold_retrieved and correct          -> "success"
    #       gold_retrieved and not correct      -> "generation_failure"   (includes a wrong abstain: the answer was RIGHT THERE)
    #       not gold_retrieved and correct      -> "suspicious"           (parametric memory or a gold label you missed)
    #       not gold_retrieved and not correct  -> "retrieval_failure"
    if value is None:
        if is_abstain(reply):
            return "correct_abstain"
        else:
            return "hallucination"
    formatted_value = f"{value:,}"
    pattern = rf"(?<![\d,]){re.escape(formatted_value)}(?![\d,])"

    correct = bool(re.search(pattern,reply)) and not is_abstain(reply)
    gold_retrieved = any(
    gold_id in retrieved_ids
    for gold_id in gold_ids
    )
    if gold_retrieved and correct:
        return "success"

    elif gold_retrieved and not correct:
        return "generation_failure"

    elif not gold_retrieved and correct:
        return "suspicious"

    else:
        return "retrieval_failure"
     

# ---------------------------------------------------------------------------
# Self-checks (fake embedder + fake LLM). Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


def _fake_embed(texts):
    # hashed bag of words: deterministic, no model. Similar word sets -> similar vectors.
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for r, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[r, sum(map(ord, w)) * 2654435761 % 256] += 1.0
    return out


if __name__ == "__main__":
    ok = []
    docs = [{"title": "Kestrel Dynamics 10-K FY2024", "text": "Kestrel Dynamics designs avionics.\n\nIn fiscal 2024, the Company's capital expenditures were $1,284 million. " * 1
             + "\n\nRisk factors include supply chain disruption and customer concentration in defense programs."},
            {"title": "Orvane Biotech 10-K FY2024", "text": "Orvane Biotech develops oncology drugs.\n\nIn fiscal 2024, the Company's research and development expense was $812 million."
             + "\n\nThe Company expects two Phase III readouts next year."},
            {"title": "Tallis Freight 10-K FY2024", "text": "Tallis Freight runs rail and trucking.\n\nIn fiscal 2024, the Company's operating income was $455 million."}]

    chunks, D = build_index(docs, _fake_embed, size=120)
    ok.append(check("1 chunks carry their doc index", len(chunks) >= 6 and {c["doc"] for c in chunks} == {0, 1, 2}
                    and all(c["text"] in docs[c["doc"]]["text"] for c in chunks), f"{len(chunks)} chunks"))
    ok.append(check("1 one unit-norm row per chunk", D.shape == (len(chunks), 256) and np.allclose(np.linalg.norm(D, axis=1), 1, atol=1e-5)))
    hchunks, _ = build_index(docs, _fake_embed, size=120, header=True)
    ok.append(check("1 header prefixes every chunk", all(c["text"].startswith("[" + docs[c["doc"]]["title"] + "] ") for c in hchunks)))

    seen = {}
    def fake_llm(messages):
        seen["messages"] = messages
        user = messages[-1]["content"]
        found = re.findall(r'<doc id="(\d+)">\n(.*?)\n</doc>', user, flags=re.S)
        for doc_id, body in found:
            if "1,284" in body:
                return f"$1,284 million [{doc_id}]"
        return "I don't have enough information."

    q = "What were Kestrel Dynamics capital expenditures in fiscal 2024?"
    out = rag_answer(q, hchunks, build_index(docs, _fake_embed, size=120, header=True)[1], lambda t: _fake_embed([t])[0], fake_llm, k=4)
    ok.append(check("2 returns the fields", set(out) >= {"reply", "retrieved", "shown", "input_tokens"} and len(out["retrieved"]) == 4 and sorted(out["shown"]) == sorted(out["retrieved"])))
    ok.append(check("2 question goes last, system first", seen["messages"][0]["content"] == SYSTEM and seen["messages"][-1]["content"].endswith(q)))
    cited = parse_citations(out["reply"])
    ok.append(check("2 shown order maps a citation to the right chunk", len(cited) == 1 and "1,284" in hchunks[out["shown"][cited[0] - 1]]["text"],
                    f"reply {out['reply']!r}, shown {out['shown']}"))
    ok.append(check("2 shown order is the edge order, not retrieval order", out["shown"] != out["retrieved"] or len(set(out["retrieved"])) < 3))

    ok.append(check("3 parse simple", parse_citations("$728 million [3].") == [3]))
    ok.append(check("3 parse many + dedupe + sort", parse_citations("see [5] and [2], also [2]") == [2, 5]))
    ok.append(check("3 parse list form", parse_citations("$1 million [1, 4]") == [1, 4]))
    ok.append(check("3 parse ignores titles / none", parse_citations("[Kestrel Dynamics 10-K FY2024] says $5") == [] and parse_citations("no cite") == []))

    sc = [{"doc": 0, "text": "capex was $1,284 million"}, {"doc": 1, "text": "R&D was $812 million"}, {"doc": 2, "text": "nothing here"}]
    shown = [2, 0, 1]  # the model saw chunk 2 as [1], chunk 0 as [2], chunk 1 as [3]
    ok.append(check("4 supported citation", citation_ok("$1,284 million [2]", 1284, shown, sc)))
    ok.append(check("4 cites the wrong doc", not citation_ok("$1,284 million [1]", 1284, shown, sc)))
    ok.append(check("4 out-of-range / missing citation", not citation_ok("$1,284 million [7]", 1284, shown, sc) and not citation_ok("$1,284 million", 1284, shown, sc)))
    ok.append(check("4 one good cite among several is enough", citation_ok("$1,284 million [1][2]", 1284, shown, sc)))

    ok.append(check("5 success", grade("$1,284 million [2]", 1284, [7], [3, 7, 9]) == "success"))
    ok.append(check("5 generation failure (wrong number)", grade("$1,150 million [2]", 1284, [7], [3, 7, 9]) == "generation_failure"))
    ok.append(check("5 generation failure (abstained with gold present)", grade("I don't have enough information.", 1284, [7], [7]) == "generation_failure"))
    ok.append(check("5 retrieval failure", grade("I don’t have enough information.", 1284, [7], [1, 2]) == "retrieval_failure"))
    ok.append(check("5 suspicious", grade("$1,284 million", 1284, [7], [1, 2]) == "suspicious"))
    ok.append(check("5 unanswerable", grade("I don't have enough information.", None, [], [1]) == "correct_abstain"
                    and grade("$300 million [1]", None, [], [1]) == "hallucination"))
    ok.append(check("5 '1,284' must be the figure, not '11,284'", grade("$11,284 million", 1284, [7], [7]) == "generation_failure"))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
