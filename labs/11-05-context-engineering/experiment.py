# Lesson 11/05 "Use it": your assembler against a real model (llama3.2 3B via Ollama, T=0).
# Run: python labs/11-05-context-engineering/experiment.py      (needs `ollama serve` + llama3.2; ~5-8 min)
# The corpus is 24 FICTIONAL company filings, so the model can't answer from memory: every correct answer
# has to come from the context you assembled. Each question asks for one figure from one company.
#   A. Silent truncation: the same assembled prompt with and without num_ctx.
#   B. Lost in the middle: the answer doc at position 1, 6, 12, 18 or 24 of 24.
#   C. How many docs: top-3 vs top-8 vs top-24 through your assemble(), answer doc ranked #2.
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_context import assemble, count_tokens  # your code

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
SYSTEM = ("You are a financial analyst assistant. Answer ONLY from the documents provided. "
          "Reply with the figure and the doc id, like: $1,234 million [3]. "
          "End every reply with the line: Source: EDGAR")

NAMES = ["Kestrel Dynamics", "Orvane Biotech", "Tallis Freight", "Marrow Semiconductor", "Quillon Foods",
         "Brightwater Utilities", "Ferro Lane Retail", "Halcyon Insurance", "Nimbus Robotics", "Corvid Mining",
         "Pellucid Health", "Saffron Hospitality", "Ironbark Steel", "Lumen Arc Energy", "Vesper Airlines",
         "Glasshouse Media", "Tundra Telecom", "Oakmere Bank", "Zephyr Apparel", "Cobalt Ridge Pharma",
         "Northgate Logistics", "Ember Software", "Pinnacle Agritech", "Wrenfield Chemicals"]
METRICS = [("total net revenue", (1500, 9800)), ("research and development expense", (120, 900)),
           ("operating income", (200, 2400)), ("capital expenditures", (90, 1100))]
FILLER = ["Management believes liquidity remains sufficient for the next twelve months.",
          "The Company operates in a highly competitive environment and faces pricing pressure.",
          "Foreign currency movements reduced reported growth by approximately one percentage point.",
          "See Note 7 for a discussion of contingencies and pending litigation.",
          "Results for the prior year have been recast to reflect the new segment structure.",
          "The effective tax rate decreased primarily due to the release of a valuation allowance."]


def make_corpus(seed=7):
    rng = random.Random(seed)
    docs, facts = [], []
    for name in NAMES:
        vals = {m: rng.randint(*r) for m, r in METRICS}
        prior = {m: int(v * rng.uniform(0.82, 0.97)) for m, v in vals.items()}  # FY2023: a built-in hard distractor
        lines = [f"{name} Form 10-K, fiscal year 2024, Item 7 (MD&A)."]
        for m, _ in METRICS:
            lines.append(f"In fiscal 2024, {m} was ${vals[m]:,} million, compared with ${prior[m]:,} million in fiscal 2023.")
            lines.append(rng.choice(FILLER))
        docs.append(" ".join(lines))
        facts.append(vals)
    return docs, facts


def ask(messages, num_ctx=None):
    opts = {"temperature": 0, "num_predict": 60}
    if num_ctx:
        opts["num_ctx"] = num_ctx
    body = {"model": MODEL, "messages": messages, "stream": False, "options": opts}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return d["message"]["content"], d.get("prompt_eval_count", 0), time.perf_counter() - t0


def correct(reply, value):
    return f"{value:,}" in reply


def questions(facts, n, seed=11):
    rng = random.Random(seed)
    targets = rng.sample(range(len(NAMES)), n)
    out = []
    for t in targets:
        metric = rng.choice(METRICS)[0]
        out.append((t, f"What was {NAMES[t]}'s fiscal 2024 {metric}?", facts[t][metric]))
    return out


def place(docs, target, position):
    # put docs[target] at 1-based `position` among all docs; others keep a fixed shuffled order
    others = [i for i in range(len(docs)) if i != target]
    random.Random(target).shuffle(others)
    order = others[:position - 1] + [target] + others[position - 1:]
    return [docs[i] for i in order]


if __name__ == "__main__":
    docs, facts = make_corpus()
    print(f"corpus: {len(docs)} fictional filings, {sum(count_tokens(d) for d in docs)} tokens total\n")

    # ---------------------------------------------------------------- A
    print("A. silent truncation (all 24 docs, window 8192)")
    t, q, v = questions(facts, 1, seed=3)[0]
    scores = [0.9 if i == t else 0.5 - i * 0.01 for i in range(len(docs))]
    msgs, rep = assemble(SYSTEM, q, docs, scores, [], window=8192, reserve=512)
    print(f"   your count: input_total {rep['input_total']} tokens, {rep['n_docs']} docs")
    for label, n_ctx in (("no num_ctx (Ollama default)", None), ("num_ctx = 8192", 8192)):
        reply, seen, secs = ask(msgs, n_ctx)
        print(f"   {label:<28} model saw {seen:>5} tokens   correct={correct(reply, v)!s:<5}  "
              f"'Source: EDGAR' rule followed={('Source: EDGAR' in reply)!s:<5}  reply: {reply.strip()[:70]!r}")

    # ---------------------------------------------------------------- B
    print("\nB. lost in the middle (answer doc among 24, num_ctx 8192, 8 questions per position)")
    qs = questions(facts, 8)
    for pos in (1, 6, 12, 18, 24):
        hits = 0
        for t, q, v in qs:
            ordered = place(docs, t, pos)
            user = "<documents>\n" + "\n".join(f'<doc id="{i + 1}">\n{d}\n</doc>' for i, d in enumerate(ordered)) + "\n</documents>\n\nQuestion: " + q
            reply, _, _ = ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], 8192)
            hits += correct(reply, v)
        print(f"   position {pos:>2}/24: {hits}/{len(qs)} correct  " + "#" * hits)

    # ---------------------------------------------------------------- C
    print("\nC. how many docs? (your assemble + reorder_edges; answer doc is retrieval rank #2, num_ctx 8192)")
    for k in (3, 8, 24):
        hits, toks, secs_all = 0, 0, 0.0
        for t, q, v in qs:
            rng = random.Random(t)
            scores = [rng.uniform(0.2, 0.6) for _ in docs]
            scores[t] = 0.80
            scores[(t + 1) % len(docs)] = 0.83  # a different company outranks the answer
            top = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
            msgs, rep = assemble(SYSTEM, q, [docs[i] for i in top], [scores[i] for i in top], [], window=8192, reserve=512)
            reply, seen, secs = ask(msgs, 8192)
            hits += correct(reply, v); toks += seen; secs_all += secs
        print(f"   top-{k:<2}: {hits}/{len(qs)} correct   avg {toks / len(qs):>5.0f} prompt tokens   avg {secs_all / len(qs):.1f} s/call")
