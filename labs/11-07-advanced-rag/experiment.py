# Lesson 11/07 "Use it": vector vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder rerank, on fictional 10-K chunks.
# Run: python labs/11-07-advanced-rag/experiment.py      (needs sentence-transformers; no LLM; ~1 min)
# 16 fictional companies x FY2024/FY2023 filings. One paragraph = one chunk, each with a "[Company 10-K FYyyyy] " header
# (the 11/06 fix). Three query types, 16 each:
#   code       : a model number or court case number, no company name      ("exact token" queries)
#   paraphrase : a risk described in DIFFERENT words than the filing        ("meaning" queries)
#   metric     : company + metric + fiscal year; the FY2023 filing is a near-duplicate distractor
# Metric: hit@1 (a gold chunk is ranked first) and hit@5 (a gold chunk is in the top 5 sent to the LLM).
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_hybrid import BM25, hybrid, rerank  # noqa: E402  (your code)
sys.path.insert(0, os.path.join(HERE, "..", "11-04-embeddings"))
from my_vector_index import exact_topk  # noqa: E402

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NAMES = ["Kestrel Dynamics", "Orvane Biotech", "Tallis Freight", "Marrow Semiconductor", "Quillon Foods",
         "Brightwater Utilities", "Ferro Lane Retail", "Halcyon Insurance", "Nimbus Robotics", "Corvid Mining",
         "Pellucid Health", "Saffron Hospitality", "Ironbark Steel", "Lumen Arc Energy", "Vesper Airlines", "Glasshouse Media"]
BUSINESS = ["designs avionics", "develops oncology drugs", "operates rail and trucking networks", "makes analog chips",
            "produces packaged foods", "supplies electricity and gas", "runs discount stores", "writes commercial insurance",
            "builds warehouse robots", "mines copper", "operates clinics", "owns resort hotels", "makes steel",
            "develops solar farms", "flies regional routes", "produces streaming content"]
PRODUCT = ["flight-control computer", "infusion therapy", "refrigerated trailer", "power-management chip", "frozen entree",
           "smart meter", "private-label appliance", "cyber liability policy", "picking robot", "ore crusher",
           "telehealth kiosk", "resort membership", "rebar coil", "battery storage unit", "turboprop charter", "documentary series"]
CLAIM = ["breach of a supply contract", "patent infringement", "a wage-and-hour class action", "trade secret misappropriation",
         "a product recall", "an environmental permit violation", "a consumer fraud claim", "bad-faith claim denial",
         "a workplace injury", "a tailings dam spill", "billing practices", "a data breach at a resort",
         "antidumping duties", "a landowner easement dispute", "a noise ordinance", "copyright infringement"]
# (risk as written in the filing, the same risk asked in other words)
RISKS = [("The Company relies on two defense ministries for most of its orders, and the loss of either customer would materially reduce sales.",
          "Which company would be badly hurt if one of its two big government buyers walked away?"),
         ("Clinical trials of the Company's lead candidates may fail to meet their endpoints, and regulators may decline to approve them.",
          "Whose main experimental medicines might not work well enough to get cleared for sale?"),
         ("Diesel prices are volatile, and the Company may be unable to pass higher fuel costs through to shippers.",
          "Which business can't always charge customers more when the cost of running its trucks goes up?"),
         ("The Company sources substantially all of its wafers from a single foundry in Taiwan, so any disruption there could halt production.",
          "Which company depends on one overseas factory to make its silicon?"),
         ("Outbreaks of foodborne illness linked to the Company's brands could trigger recalls and lasting reputational harm.",
          "Whose customers could get sick from its meals, forcing products off shelves?"),
         ("Severe storms and wildfires can damage the Company's transmission lines, and regulators may not allow recovery of repair costs.",
          "Which company might not be allowed to bill customers for fixing power lines after extreme weather?"),
         ("Shoplifting and organized retail crime have increased inventory shrink at the Company's stores.",
          "Which company is losing more merchandise to theft?"),
         ("Catastrophe losses from hurricanes may exceed the Company's reinsurance protection.",
          "Whose payouts after a major storm could be larger than the coverage it bought for itself?"),
         ("Customers may delay automation projects when interest rates are high, lengthening the Company's sales cycle.",
          "Which firm's buyers postpone purchases when borrowing gets expensive?"),
         ("The Company's largest pit is in a jurisdiction where the government has proposed raising royalties and nationalizing concessions.",
          "Which company could have its main dig site taken over by a foreign state?"),
         ("A shortage of licensed nurses has increased the Company's labor costs and forced some sites to reduce hours.",
          "Who is struggling to hire enough qualified medical staff to stay open?"),
         ("Travel demand at the Company's properties is seasonal, and a weak winter season can significantly lower annual results.",
          "Whose yearly earnings depend heavily on how busy a few cold months are?"),
         ("Imports of low-priced foreign product have depressed selling prices in the Company's core markets.",
          "Which company is being undercut by cheap goods shipped in from abroad?"),
         ("Changes to net metering rules could reduce the value of electricity the Company sells back to utilities.",
          "Whose income could shrink if the rules for crediting rooftop power exported to the grid change?"),
         ("A shortage of qualified pilots has forced the Company to cancel flights and pay higher wages to cockpit crews.",
          "Which company is cutting trips because it can't find enough people to fly its planes?"),
         ("Subscriber churn may rise if the Company's content library fails to attract viewers.",
          "Whose members might cancel if its shows and films stop drawing an audience?")]


def make_corpus(seed=7):
    rng = random.Random(seed)
    chunks, qs = [], []
    for c, name in enumerate(NAMES):
        v = {}
        for m in ("total net revenue", "operating income", "research and development expense", "capital expenditures"):
            v[(2024, m)] = rng.randint(300, 9800)
            v[(2023, m)] = int(v[(2024, m)] * rng.uniform(0.82, 0.97))
            v[(2022, m)] = int(v[(2023, m)] * rng.uniform(0.82, 0.97))
        code = f"{name[0]}{name.split()[1][0]}-{rng.randint(1000, 9999)}"
        case = f"{rng.randint(1, 4)}:{rng.choice((23, 24))}-cv-{rng.randint(1000, 99999):05d}"
        ids = {}
        for fy in (2024, 2023):
            s = lambda m: f"In fiscal {fy}, {m} was ${v[(fy, m)]:,} million, compared with ${v[(fy - 1, m)]:,} million in fiscal {fy - 1}."
            paras = {"intro": f"{name} (the \"Company\") {BUSINESS[c]}. This Annual Report on Form 10-K covers the fiscal year ended December 31, {fy}.",
                     "mdna": "Management's Discussion and Analysis. " + s("total net revenue") + " " + s("operating income"),
                     "invest": "Investment. " + s("research and development expense") + " " + s("capital expenditures"),
                     "product": f"Products. The Company's principal {PRODUCT[c]} is sold under model number {code}. Unit shipments of the {code} changed {rng.randint(-9, 25)}% in fiscal {fy}.",
                     "legal": f"Legal Proceedings. The Company is a defendant in case No. {case}, which alleges {CLAIM[c]}. The Company believes the claims are without merit.",
                     "risk": "Risk Factors. " + RISKS[c][0]}
            for kind, text in paras.items():
                ids[(fy, kind)] = len(chunks)
                chunks.append(f"[{name} 10-K FY{fy}] " + text)
        both = lambda kind: [ids[(2024, kind)], ids[(2023, kind)]]
        qs.append({"type": "code", "q": f"Which filing discusses shipments of model {code}?" if c % 2 == 0
                   else f"What does case No. {case} allege?", "gold": both("product" if c % 2 == 0 else "legal")})
        qs.append({"type": "paraphrase", "q": RISKS[c][1], "gold": both("risk")})
        fy, m = rng.choice((2024, 2023)), rng.choice(("research and development expense", "capital expenditures", "operating income"))
        qs.append({"type": "metric", "q": f"What was {name}'s {m} in fiscal {fy}?",
                   "gold": [ids[(fy, "invest" if m != "operating income" else "mdna")]]})
    return chunks, qs


if __name__ == "__main__":
    chunks, qs = make_corpus()
    print(f"{len(chunks)} chunks, {len(qs)} queries (16 code, 16 paraphrase, 16 metric)")
    st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    D = st.encode(chunks, normalize_embeddings=True, batch_size=64).astype(np.float32)
    bm = BM25().fit(chunks)
    cross = lambda pairs: ce.predict(pairs, batch_size=64)

    def vector(q, qv):
        ids, _ = exact_topk(qv.reshape(1, -1), D, 50)
        return [int(i) for i in np.asarray(ids).reshape(-1)]

    methods = {"vector": vector,
               "BM25": lambda q, qv: bm.topk(q, 50),
               "hybrid (RRF)": lambda q, qv: hybrid(q, qv, D, bm, n=50),
               "hybrid + rerank": lambda q, qv: rerank(q, hybrid(q, qv, D, bm, n=50)[:50], chunks, cross, top_k=5)}

    qvecs = st.encode([QUERY_PREFIX + q["q"] for q in qs], normalize_embeddings=True).astype(np.float32)
    results, secs = {}, {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        results[name] = [fn(q["q"], qv) for q, qv in zip(qs, qvecs)]
        secs[name] = (time.perf_counter() - t0) / len(qs) * 1000

    types = ("code", "paraphrase", "metric")
    print(f"\n{'':<17}" + "".join(f"{t:>18}" for t in types) + f"{'all':>12}{'ms/query':>10}")
    print(f"{'':<17}" + "".join(f"{'hit@1   hit@5':>18}" for _ in types))
    for name in methods:
        row = f"{name:<17}"
        for t in types:
            idx = [i for i, q in enumerate(qs) if q["type"] == t]
            h1 = sum(results[name][i][0] in qs[i]["gold"] if results[name][i] else 0 for i in idx)
            h5 = sum(any(g in results[name][i][:5] for g in qs[i]["gold"]) for i in idx)
            row += f"{h1:>10}/16 {h5:>2}/16"
        a1 = sum(bool(r) and r[0] in q["gold"] for r, q in zip(results[name], qs))
        row += f"{a1:>9}/48{secs[name]:>9.1f}"
        print(row)

    print("\nWhere each method's #1 went wrong (first 2 misses per type):")
    for name in methods:
        for t in types:
            miss = [i for i, q in enumerate(qs) if q["type"] == t and not (results[name][i] and results[name][i][0] in q["gold"])][:2]
            for i in miss:
                top = chunks[results[name][i][0]][:80] if results[name][i] else "(no keyword match at all)"
                print(f"  {name:<16} {t:<10} {qs[i]['q'][:55]!r:<58} -> {top!r}")
