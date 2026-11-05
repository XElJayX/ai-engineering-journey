# Lesson 11/05 Context Engineering: a token-budgeted context assembler.
# Run: python labs/11-05-context-engineering/my_context.py     (needs tiktoken; self-checks only, no LLM)
# The "Use it" run (experiment.py) sends your assembled context to llama3.2 and measures truncation and
# lost-in-the-middle. Token counts use tiktoken's cl100k_base as a stand-in for llama3's tokenizer
# (both are BPE with ~100K+ vocab; budget with a margin, never with words * 1.3, see 05/19).
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
MSG_OVERHEAD = 4  # chat templates add role/header tokens per message (llama3: <|start_header_id|>role<|end_header_id|>\n\n ... <|eot_id|>)
DOC_OVERHEAD = 12  # upper bound on the tokens render_docs adds per chunk: <doc id="12">\n ... \n</doc>\n


def count_tokens(text):
    return len(ENC.encode(text)) if text else 0


class ContextOverflow(Exception):
    pass


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def reorder_edges(items, scores):
    # TODO 1: lost-in-the-middle reordering. Rank items by score, best first (ties keep their original order:
    # use a stable sort). Then put rank 0 first, rank 1 LAST, rank 2 second, rank 3 second-to-last, ...
    # so the strongest items sit at both edges and the weakest end up in the middle.
    #   5 items ranked r0..r4  ->  [r0, r2, r4, r3, r1]
    # Return the reordered list of items.
    pairs = list(zip(items,scores))
    sorted_pairs = sorted(pairs, key = lambda x: x[1], reverse = True )

    left = []
    right = []

    for i in range(len(pairs)):
        if i%2 == 0:
            left.append(sorted_pairs[i][0])
        else:
            right.append(sorted_pairs[i][0])
    right = right[::-1]
    return left + right



def select_docs(docs, scores, budget, max_docs=None, per_doc_overhead=0):
    # TODO 2: choose which retrieved chunks go in. Walk the docs in score order (best first, stable).
    #   - A chunk costs count_tokens(doc) + per_doc_overhead (the tags around it are tokens too).
    #   - Never 
    # split a chunk (half a table is worse than no table).
    #   - If a chunk doesn't fit the remaining budget, SKIP it and keep going: a later, smaller chunk may fit.
    #   - Stop once max_docs chunks are selected (None = no limit).
    # Return (selected_docs, selected_scores, tokens_used), all in score order.
    pairs = list(zip(docs,scores))
    sorted_pairs = sorted(pairs, key = lambda x: x[1], reverse = True)

    selected_docs = []
    selected_scores = []
    tokens_used = 0

    for doc,score in sorted_pairs:
        cost = count_tokens(doc) + per_doc_overhead
        if max_docs is not None and len(selected_docs) >= max_docs:
            break
        if cost + tokens_used <= budget:
            selected_docs.append(doc)
            selected_scores.append(score)
            tokens_used += cost
            
    return(selected_docs, selected_scores, tokens_used)






def window_history(turns, budget):
    # TODO 3: sliding-window history. turns = [{"role": "user"|"assistant", "content": str}, ...], oldest first.
    # Walk from the NEWEST turn backwards, keeping whole turns (each costs count_tokens(content) + MSG_OVERHEAD)
    # while they fit. At the first turn that doesn't fit, STOP (unlike docs, don't skip ahead: a conversation
    # with a hole in the middle is incoherent).
    # Return (kept, dropped), both oldest first. kept + dropped == turns (dropped is the older prefix).
    kept = []

    token_used = 0
    for turn in reversed(turns):
        cost = count_tokens(turn["content"]) + MSG_OVERHEAD
        if cost + token_used > budget:
            break
        token_used += cost
        kept.append(turn)
    kept = kept[::-1]
    n_dropped = len(turns) - len(kept)
    dropped = turns[:n_dropped]
    return (kept,dropped)


def render_docs(docs):
    # Given. Wrap chunks in tags so the model (and a prompt-injection filter later) can tell data from instructions.
    return "<documents>\n" + "\n".join(f'<doc id="{i + 1}">\n{d}\n</doc>' for i, d in enumerate(docs)) + "\n</documents>"


def assemble(system, query, docs, scores, history, window, reserve, doc_cap=None):
    # TODO 4: build the messages list for one call. Priority and layout:
    #   1. FIXED parts: system message, the final user message's question text, and the generation reserve.
    #      fixed = count_tokens(system) + MSG_OVERHEAD + count_tokens(question_text) + MSG_OVERHEAD + reserve
    #      where question_text = "Question: " + query
    #      If fixed > window, raise ContextOverflow (never cut the system prompt or the query).
    #   2. DOCS get the remaining budget, capped at doc_cap if given: select_docs(..., budget, per_doc_overhead=DOC_OVERHEAD).
    #      The <documents> wrapper costs tokens too: budget = available - count_tokens(render_docs([])) - 2,
    #      and if no docs are selected, don't include the wrapper at all.
    #   3. HISTORY gets whatever is left after docs: window_history(history, left).
    #   4. Layout: [system] + kept history turns + [final user message]
    #      final user content = render_docs(reorder_edges(selected, selected_scores)) + "\n\n" + question_text
    #      (no docs selected -> final user content is just question_text). The question goes LAST.
    # Return (messages, report) where report is a dict with integer token counts:
    #   {"system", "docs", "history", "query", "reserve", "input_total", "n_docs", "n_turns", "n_dropped_turns"}
    #   input_total = sum(count_tokens(m["content"]) + MSG_OVERHEAD for m in messages); it must be <= window - reserve.
    question_text = "Question: " + query
    fixed = count_tokens(system) + MSG_OVERHEAD+ count_tokens(question_text) + MSG_OVERHEAD + reserve
    if fixed > window:
        raise ContextOverflow(
        f"input total {total} exceeds budget {window - reserve}"
    )
    available = window - fixed
    doc_budget = max(0, available - count_tokens(render_docs([])) - 2)

    if doc_cap is not None:
        doc_budget = min(doc_cap,doc_budget)
    selected, selected_scores, tokens_used = select_docs(
    docs,
    scores,
    budget=doc_budget,
    per_doc_overhead=DOC_OVERHEAD
    )
    reordered = reorder_edges(selected, selected_scores)
    if len(reordered) > 0 :
        doc_text = render_docs(reordered)
    else:
        doc_text = ""
    doc_cost = count_tokens(doc_text)
    history_budget= available - doc_cost
    kept,dropped = window_history(history,budget = history_budget)

    if doc_text:
        final_user_content = doc_text + "\n\n" + question_text
    else:
        final_user_content = question_text
    
    messages = [
    {"role": "system", "content": system}
    ]
    messages.extend(kept)
    messages.append({
    "role": "user",
    "content": final_user_content
    })
    total = sum(count_tokens(m["content"]) + MSG_OVERHEAD for m in messages)
    while total > window - reserve and len(kept) > 0:
    # Remove oldest retained history turn
        removed = kept.pop(0)

    # Move it to the dropped history
        dropped.append(removed)
        messages = [{"role": "system", "content": system}]
        messages.extend(kept)
        messages.append({"role": "user", "content": final_user_content})
        total = sum(count_tokens(m["content"]) + MSG_OVERHEAD for m in messages)

    while total > window - reserve and len(selected) > 0:
        selected.pop()
        selected_scores.pop()

        reordered = reorder_edges(selected, selected_scores)

        doc_text = render_docs(reordered) if reordered else ""

        final_user_content = doc_text + "\n\n" + question_text if doc_text else question_text
        messages = [{"role": "system", "content": system}]
        messages.extend(kept)
        messages.append({"role": "user", "content": final_user_content})

        total = sum(
            count_tokens(m["content"]) + MSG_OVERHEAD
            for m in messages
        )
    query_cost = count_tokens(question_text) + MSG_OVERHEAD
    final_user_cost = count_tokens(final_user_content) + MSG_OVERHEAD
    docs_cost = final_user_cost - query_cost

    report = {
    "system": count_tokens(system) + MSG_OVERHEAD,
    "docs": docs_cost,
    "history": sum(
    count_tokens(turn["content"]) + MSG_OVERHEAD
    for turn in kept
    ),
    "query":query_cost,
    "reserve": reserve,
    "input_total": total,
    "n_docs":len(reordered),
    "n_turns": len(kept),
    "n_dropped_turns": len(dropped)
    }
    return messages, report




# ---------------------------------------------------------------------------
# Self-checks. Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


if __name__ == "__main__":
    ok = []
    table = "| Segment | FY2024 | FY2023 |\n| iPhone | $201,183 | $200,583 |\n| Services | $96,169 | $85,200 |"
    words = len(table.split()) * 1.3
    print(f"(why not words*1.3: the table is {count_tokens(table)} tokens, words*1.3 says {words:.0f})\n")

    ok.append(check("1 reorder 5", reorder_edges(list("ABCDE"), [0.9, 0.7, 0.5, 0.3, 0.1]) == list("ACEDB"),
                    str(reorder_edges(list("ABCDE"), [0.9, 0.7, 0.5, 0.3, 0.1]))))
    ok.append(check("1 reorder unsorted input", reorder_edges(list("ABCD"), [0.1, 0.9, 0.5, 0.7]) == list("BCAD"),
                    str(reorder_edges(list("ABCD"), [0.1, 0.9, 0.5, 0.7]))))
    ok.append(check("1 reorder ties stable + tiny", reorder_edges(list("XYZ"), [0.5, 0.5, 0.5]) == list("XZY")
                    and reorder_edges(["a"], [1]) == ["a"] and reorder_edges([], []) == []))

    big, small = "word " * 300, "word " * 50
    sel, sc, used = select_docs([big, small, big], [0.9, 0.8, 0.7], budget=400)
    ok.append(check("2 select skips what doesn't fit", sel == [big, small] and sc == [0.9, 0.8] and used == count_tokens(big) + count_tokens(small),
                    f"got {len(sel)} docs, {used} tokens"))
    sel, sc, used = select_docs([small, big, small], [0.2, 0.9, 0.5], budget=360)
    ok.append(check("2 select is score-ordered, skip-and-continue", sel == [big, small] and sc == [0.9, 0.5],
                    f"scores {sc}"))
    sel, _, _ = select_docs([small] * 5, [0.5] * 5, budget=10_000, max_docs=2)
    ok.append(check("2 select max_docs", len(sel) == 2))

    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "blah " * (20 if i != 3 else 200)} for i in range(6)]
    cost = [count_tokens(t["content"]) + MSG_OVERHEAD for t in turns]
    kept, dropped = window_history(turns, budget=cost[4] + cost[5] + 5)
    ok.append(check("3 history keeps newest whole turns", kept == turns[4:] and dropped == turns[:4]))
    kept, dropped = window_history(turns, budget=cost[4] + cost[5] + cost[0] + cost[1] + cost[2] + 5)
    ok.append(check("3 history stops at first misfit (no hole)", kept == turns[4:] and dropped == turns[:4],
                    f"kept turns {[t['content'][:6] for t in kept]}"))
    kept, dropped = window_history(turns, budget=10**6)
    ok.append(check("3 history all fit / order", kept == turns and dropped == []))

    system = "You are a financial analyst assistant. Answer only from the documents. Cite doc ids like [2]."
    query = "What were Apple's FY2024 Services net sales?"
    docs = [f"Chunk {i}: " + "revenue " * 80 for i in range(10)]
    scores = [0.3, 0.8, 0.5, 0.9, 0.1, 0.6, 0.7, 0.2, 0.4, 0.05]
    msgs, rep = assemble(system, query, docs, scores, turns, window=1200, reserve=300)
    ok.append(check("4 layout system first, question last", msgs[0]["role"] == "system" and msgs[0]["content"] == system
                    and msgs[-1]["role"] == "user" and msgs[-1]["content"].endswith("Question: " + query)))
    ok.append(check("4 fits window - reserve", rep["input_total"] <= 1200 - 300 and rep["reserve"] == 300,
                    f"input_total {rep.get('input_total')}"))
    real_total = sum(count_tokens(m["content"]) + MSG_OVERHEAD for m in msgs)
    ok.append(check("4 report is honest", rep["input_total"] == real_total, f"report {rep.get('input_total')} vs real {real_total}"))
    ok.append(check("4 best doc at the front edge", '<doc id="1">\nChunk 3:' in msgs[-1]["content"], "doc 1 should be the 0.9 chunk"))
    ok.append(check("4 docs before history (priority)", rep["n_docs"] >= 7, f"n_docs {rep.get('n_docs')}"))
    msgs2, rep2 = assemble(system, query, docs, scores, turns, window=1200, reserve=300, doc_cap=250)
    ok.append(check("4 doc_cap frees room for history", rep2["n_docs"] < rep["n_docs"] and rep2["n_turns"] > rep["n_turns"],
                    f"docs {rep2.get('n_docs')} turns {rep2.get('n_turns')}"))
    msgs3, rep3 = assemble(system, query, [], [], [], window=1200, reserve=300)
    ok.append(check("4 no docs -> no empty wrapper", msgs3[-1]["content"] == "Question: " + query and len(msgs3) == 2))
    try:
        assemble(system * 60, query, docs, scores, turns, window=1200, reserve=300)
        ok.append(check("4 overflow raises", False, "no exception"))
    except ContextOverflow:
        ok.append(check("4 overflow raises", True))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
