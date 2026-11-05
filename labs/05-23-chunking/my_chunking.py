# Bridge lesson 11: Chunking Strategies for RAG. Build fixed, recursive and parent-child chunkers.
# Run: python3 labs/05-23-chunking/my_chunking.py   (stdlib only)
# Sizes are in characters here to keep it simple; production chunkers usually count tokens (05/19).

CONTRACT = """SERVICES AGREEMENT

1. Parties. This Agreement is made between Northwind Labs Inc. ("Provider") and Contoso Retail LLC ("Client").

2. Notices Address. All notices must be sent to the addresses listed on the signature page, or to any new address a party provides in writing.

3. Services. Provider will host, monitor and maintain the Client's recommendation API as described in Schedule A.

4. Fees. Client will pay a monthly fee of $12,000, invoiced on the first business day of each month.

5. Payment Terms. Invoices are due within 30 days. Late payments accrue interest of 1.5% per month.

6. Service Levels. Provider guarantees 99.9% monthly uptime. If uptime falls below this level, Client receives a credit of 10% of that month's fee.

7. Data Protection. Provider will process Client data only to deliver the Services and will delete it within 60 days after the Agreement ends.

8. Confidentiality. Each party will keep the other party's confidential information secret for five years after disclosure.

9. Intellectual Property. Client owns its data and models. Provider owns its tooling and any general improvements to it.

10. Liability. Each party's total liability is capped at the fees paid in the twelve months before the claim.

11. Term. This Agreement starts on March 1, 2026 and renews automatically for one-year periods.

12. Termination.
12.1 Either party may terminate this Agreement with 30 days' written notice.
12.2 Such notice must be sent by certified mail to the address in Section 2.
12.3 Client may terminate immediately if uptime falls below 95% in two consecutive months.

13. Governing Law. This Agreement is governed by the laws of the State of Delaware.
"""


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def chunk_fixed(text, size, overlap=0):
    # Cut every `size` characters; consecutive chunks share `overlap` characters (step = size - overlap).
    # Raise ValueError if size <= 0 or overlap < 0 or overlap >= size.
    # Don't emit a final chunk that lies entirely inside the previous one:
    #   chunk_fixed("abcdefghij", 4, 1) -> ["abcd", "defg", "ghij"]   (not a 4th chunk "j")
    # Empty text -> [].
    if size <=0 or overlap>=size or overlap < 0:
        raise ValueError
    if text =="":
        return []
    chunks=[]
    step = size - overlap
    l,r = 0, len(text)
    while l<r:
        chunks.append(text[l:l+size])
        if l+size >= r:
            break
        l+=step
    return chunks


def chunk_recursive(text, size, seps=("\n\n", "\n", ". ", " ")):
    # 1. If the whole text fits (len <= size), return [text].
    # 2. Otherwise take the FIRST separator in `seps` that occurs in the text and split on it.
    #    KEEP the separator on the end of each piece (so no characters are lost): "a. b. c" on ". " -> ["a. ", "b. ", "c"].
    # 3. Greedily pack consecutive pieces into a buffer while it stays <= size; when the next piece doesn't fit,
    #    emit the buffer and start a new one.
    # 4. A single piece that is itself > size: emit the current buffer, then chunk that piece recursively
    #    with the REMAINING separators (seps after the current one).
    # 5. If no separator occurs at all, fall back to chunk_fixed(text, size).
    # Invariant the checks test: "".join(chunks) == text, and every chunk has len <= size.
    if len(text) <= size:
        return [text]

    for sep in seps:
        if sep in text:
            pieces = text.split(sep)
            pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]]
            chunks = []
            buffer = ""
            for piece in pieces:
                if len(buffer) + len(piece) <= size:
                    buffer += piece
                else:
                    if buffer:
                        chunks.append(buffer)
                    if len(piece) > size:
                        chunks.extend(chunk_recursive(piece, size, seps[seps.index(sep)+1:]))
                        buffer = ""
                    else:
                        buffer = piece
            if buffer:
                chunks.append(buffer)
            return chunks
    return chunk_fixed(text, size)



def parent_child(text, parent_size, child_size):
    # Chunk the text into parents with chunk_recursive(text, parent_size), then chunk EACH parent into children
    # with chunk_recursive(parent, child_size). Return a list of dicts, one per child:
    #   {"child": <child text>, "parent_idx": <index of its parent>, "parent": <parent text>}
    parents = chunk_recursive(text, parent_size)
    result = []
    for i, parent in enumerate(parents):
        children = chunk_recursive(parent, child_size)
        for child in children:
            result.append({"child": child, "parent_idx": i, "parent": parent})
    return result


def retrieve_parents(mapping, child_scores, top_k):
    # mapping: output of parent_child. child_scores: one similarity score per entry of mapping.
    # Take the top_k CHILDREN by score (highest first), and return their parents' texts, best first,
    # WITHOUT duplicates (several children can share one parent).
    scored_children = sorted(zip(child_scores, mapping), key=lambda x: -x[0])
    top_children = scored_children[:top_k]
    seen_parents = set()
    result = []
    for score, entry in top_children:
        parent_text = entry["parent"]
        if parent_text not in seen_parents:
            seen_parents.add(parent_text)
            result.append(parent_text)
    return result


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below this line; make them pass.
# ---------------------------------------------------------------------------

def check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, "not implemented yet"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def t_fixed():
    a = chunk_fixed("abcdefghij", 4)
    b = chunk_fixed("abcdefghij", 4, 1)
    bad = 0
    for args in ((4, 4), (0, 0), (4, -1)):
        try:
            chunk_fixed("abc", *args)
        except ValueError:
            bad += 1
    ok = a == ["abcd", "efgh", "ij"] and b == ["abcd", "defg", "ghij"] and chunk_fixed("", 4) == [] and bad == 3
    return ok, f"no overlap {a}; overlap 1 {b}; rejects bad sizes {bad}/3"


def t_fixed_cuts_badly():
    cuts = chunk_fixed(CONTRACT, 400)
    sec12 = CONTRACT[CONTRACT.index("12. Termination."):CONTRACT.index("13. Governing")].strip()
    split = not any(sec12 in c for c in cuts)
    broken = [c for c in cuts[:-1] if c[-1].isalnum()]
    where = next((c for c in cuts if "12. Termination." in c), "")
    return split and len(broken) >= 1, (f"{len(cuts)} chunks, {len(broken)} end mid-word; section 12 split across chunks: {split} "
                                        f"(its chunk ends ...{where[-30:]!r})")


def t_recursive_invariants():
    for size in (60, 150, 300, 800):
        ch = chunk_recursive(CONTRACT, size)
        if "".join(ch) != CONTRACT or max(map(len, ch)) > size:
            return False, f"size {size}: rejoin ok={''.join(ch) == CONTRACT}, longest={max(map(len, ch))}"
    return True, "sizes 60/150/300/800: every chunk fits and \"\".join(chunks) == text (nothing lost)"


def t_recursive_structure():
    ch = chunk_recursive(CONTRACT, 300)
    sec12 = CONTRACT[CONTRACT.index("12. Termination."):CONTRACT.index("13. Governing")].strip()
    whole = any(sec12 in c for c in ch)
    small = chunk_recursive("one two three four five six seven", 10)
    clean = all(c.endswith(" ") for c in small[:-1])
    tight = chunk_recursive(CONTRACT, 60)
    words_intact = all(c[-1].isspace() for c in tight)
    return whole and clean and words_intact, (f"section 12 kept in one chunk: {whole}; {small}; "
                                              f"size 60: all {len(tight)} chunks end at whitespace: {words_intact}")


def t_recursive_fallback():
    ch = chunk_recursive("x" * 25, 10)
    return ch == ["x" * 10, "x" * 10, "x" * 5], f"25 chars, no separators, size 10 -> {[len(c) for c in ch]}"


def t_parent_child():
    m = parent_child(CONTRACT, 400, 120)
    ok = all(e["child"] in e["parent"] for e in m)
    by_parent = {}
    for e in m:
        by_parent.setdefault(e["parent_idx"], []).append(e["child"])
    ok = ok and all("".join(kids) == m[[e["parent_idx"] for e in m].index(i)]["parent"] for i, kids in by_parent.items())
    ok = ok and [e["parent_idx"] for e in m] == sorted(e["parent_idx"] for e in m)
    return ok, f"{len(by_parent)} parents, {len(m)} children; each parent's children rejoin to the parent exactly"


def t_dedupe():
    m = [{"child": "a1", "parent_idx": 0, "parent": "A"}, {"child": "b1", "parent_idx": 1, "parent": "B"},
         {"child": "a2", "parent_idx": 0, "parent": "A"}, {"child": "c1", "parent_idx": 2, "parent": "C"}]
    got = retrieve_parents(m, [0.9, 0.5, 0.8, 0.1], top_k=3)
    return got == ["A", "B"], f"top-3 children a1, a2, b1 -> parents {got} (A once, best first)"


if __name__ == "__main__":
    tests = [
        ("chunk_fixed: sizes, overlap, no redundant tail, rejects bad args", t_fixed),
        ("chunk_fixed on the contract cuts mid-word (the baseline's flaw)", t_fixed_cuts_badly),
        ("chunk_recursive: every chunk fits, nothing lost", t_recursive_invariants),
        ("chunk_recursive: keeps sections whole, cuts at spaces", t_recursive_structure),
        ("chunk_recursive: falls back to fixed when no separator", t_recursive_fallback),
        ("parent_child: children tile their parents", t_parent_child),
        ("retrieve_parents: top children -> unique parents", t_dedupe),
    ]
    passed = sum(check(name, fn) for name, fn in tests)
    print(f"\n  {passed}/{len(tests)} checks passed")
