# Lesson 11/06 "Use it": your RAG pipeline on fictional 10-Ks with real models (bge-small + llama3.2 3B, T=0).
# Run: python labs/11-06-rag/experiment.py      (needs sentence-transformers and `ollama serve`; ~3-5 min)
# 16 fictional companies x 2 fiscal years = 32 filings. Like real 10-Ks, most paragraphs say "the Company" and
# name it only in the opening paragraph. 24 questions: 20 answerable, 4 not answerable from the corpus.
#   closed-book   : same system prompt, no documents (what does the model do with nothing?)
#   RAG, plain    : your chunks as-is
#   RAG, header   : every chunk prefixed with "[<Company> 10-K FY<year>]" before embedding
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_rag import SYSTEM, build_index, citation_ok, grade, is_abstain, rag_answer  # your code

from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL = "llama3.2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NAMES = ["Kestrel Dynamics", "Orvane Biotech", "Tallis Freight", "Marrow Semiconductor", "Quillon Foods",
         "Brightwater Utilities", "Ferro Lane Retail", "Halcyon Insurance", "Nimbus Robotics", "Corvid Mining",
         "Pellucid Health", "Saffron Hospitality", "Ironbark Steel", "Lumen Arc Energy", "Vesper Airlines", "Glasshouse Media"]
BUSINESS = ["designs avionics", "develops oncology drugs", "operates rail and trucking networks", "makes analog chips",
            "produces packaged foods", "supplies electricity and gas", "runs discount stores", "writes commercial insurance",
            "builds warehouse robots", "mines copper", "operates clinics", "owns resort hotels", "makes steel",
            "develops solar farms", "flies regional routes", "produces streaming content"]
METRICS = [("total net revenue", (1500, 9800)), ("operating income", (200, 2400)),
           ("research and development expense", (120, 900)), ("capital expenditures", (90, 1100))]
FILLER = ["Management believes liquidity remains sufficient for the next twelve months.",
          "Foreign currency movements reduced reported growth by approximately one percentage point.",
          "See Note 7 for a discussion of contingencies and pending litigation.",
          "The effective tax rate decreased primarily due to the release of a valuation allowance.",
          "The Company faces intense competition and pricing pressure in its principal markets.",
          "Results for the prior year have been recast to reflect the new segment structure."]


def make_corpus(seed=7):
    rng = random.Random(seed)
    docs, truth = [], {}
    for c, name in enumerate(NAMES):
        v = {}
        for m, r in METRICS:
            v[(2024, m)] = rng.randint(*r)
            v[(2023, m)] = int(v[(2024, m)] * rng.uniform(0.82, 0.97))
            v[(2022, m)] = int(v[(2023, m)] * rng.uniform(0.82, 0.97))
        for fy in (2024, 2023):
            s = lambda m: f"In fiscal {fy}, {m} was ${v[(fy, m)]:,} million, compared with ${v[(fy - 1, m)]:,} million in fiscal {fy - 1}."
            paras = [f"{name} (the \"Company\") {BUSINESS[c]}. This Annual Report on Form 10-K covers the fiscal year ended December 31, {fy}.",
                     "Item 7. Management's Discussion and Analysis. " + s(METRICS[0][0]) + " " + s(METRICS[1][0]) + " " + rng.choice(FILLER),
                     "Investment. " + s(METRICS[2][0]) + " " + s(METRICS[3][0]) + " " + rng.choice(FILLER),
                     "Item 1A. Risk Factors. " + " ".join(rng.sample(FILLER, 3)),
                     "Liquidity and Capital Resources. " + " ".join(rng.sample(FILLER, 2))]
            docs.append({"title": f"{name} 10-K FY{fy}", "text": "\n\n".join(paras), "company": c, "fy": fy})
        truth[c] = v
    return docs, truth


def make_questions(truth, seed=11):
    rng = random.Random(seed)
    qs = []
    for _ in range(20):
        c, fy, m = rng.randrange(len(NAMES)), rng.choice((2024, 2023)), rng.choice(METRICS)[0]
        qs.append({"q": f"What was {NAMES[c]}'s {m} in fiscal {fy}?", "company": c, "value": truth[c][(fy, m)]})
    qs += [{"q": "What was Vantor Systems' total net revenue in fiscal 2024?", "company": None, "value": None},
           {"q": "What was Bellhaven Textiles' operating income in fiscal 2023?", "company": None, "value": None},
           {"q": f"What dividends per share did {NAMES[3]} pay in fiscal 2024?", "company": 3, "value": None},
           {"q": f"What was {NAMES[9]}'s free cash flow in fiscal 2024?", "company": 9, "value": None}]
    return qs


def llm(messages):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0, "num_predict": 80, "num_ctx": 8192}}  # 11/05: always set the window
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"]


def gold_ids(chunks, docs, q):
    if q["value"] is None:
        return []
    pat = re.compile(r"(?<![\d,])" + re.escape(f"{q['value']:,}") + r"(?![\d]|,\d)")
    return [i for i, ch in enumerate(chunks) if docs[ch["doc"]]["company"] == q["company"] and pat.search(ch["text"])]


def made_up_figure(row):
    # a $ figure that is NOT the right answer: wrong number, or any number for an unanswerable question
    wrong = row["bucket"] in ("generation_failure", "retrieval_failure", "hallucination")
    return wrong and (not is_abstain(row["reply"])) and re.search(r"\$\s?\d", row["reply"]) is not None


def report(name, rows):
    b = Counter(r["bucket"] for r in rows)
    ans = [r for r in rows if r["value"] is not None]
    correct = [r for r in ans if r["bucket"] in ("success", "suspicious")]
    cite = sum(r["cite_ok"] for r in correct)
    print(f"\n{name}")
    print(f"   answerable correct {len(correct)}/{len(ans)}   citation ok on correct {cite}/{len(correct)}   "
          f"unanswerable abstained {b['correct_abstain']}/{len(rows) - len(ans)}   wrong $ figure stated {sum(made_up_figure(r) for r in rows)}/{len(rows)}")
    print("   buckets: " + ", ".join(f"{k} {b[k]}" for k in ("success", "generation_failure", "retrieval_failure", "suspicious", "correct_abstain", "hallucination")))
    if rows and "tokens" in rows[0]:
        print(f"   avg input tokens {sum(r['tokens'] for r in rows) / len(rows):.0f}   avg {sum(r['secs'] for r in rows) / len(rows):.1f} s/question")
    for r in [r for r in rows if r["bucket"] in ("generation_failure", "hallucination", "suspicious")][:3]:
        print(f"   e.g. {r['bucket']:<18} {r['q'][:60]!r} -> {r['reply'].strip()[:70]!r}")


if __name__ == "__main__":
    docs, truth = make_corpus()
    qs = make_questions(truth)
    st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embed_docs = lambda texts: st.encode(texts, normalize_embeddings=True, batch_size=64)
    embed_query = lambda t: st.encode(QUERY_PREFIX + t, normalize_embeddings=True)
    print(f"{len(docs)} filings, {len(qs)} questions (20 answerable, 4 not)")

    rows = []
    for q in qs:
        reply = llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Question: " + q["q"]}])
        rows.append({**q, "reply": reply, "bucket": grade(reply, q["value"], [0], []), "cite_ok": False})
    report("closed-book (no documents)", rows)

    for label, header in (("RAG k=5, plain chunks", False), ("RAG k=5, header chunks", True)):
        chunks, D = build_index(docs, embed_docs, size=600, header=header)
        rows = []
        for q in qs:
            t0 = time.perf_counter()
            out = rag_answer(q["q"], chunks, D, embed_query, llm, k=5)
            g = gold_ids(chunks, docs, q)
            rows.append({**q, "reply": out["reply"], "bucket": grade(out["reply"], q["value"], g, out["retrieved"]),
                         "cite_ok": q["value"] is not None and citation_ok(out["reply"], q["value"], out["shown"], chunks),
                         "tokens": out["input_tokens"], "secs": time.perf_counter() - t0})
        report(f"{label}  ({len(chunks)} chunks)", rows)
