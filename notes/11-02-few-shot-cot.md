# 11/02 Few-Shot, Chain-of-Thought, Tree-of-Thought

> Examples teach format. Reasoning tokens give the model room to compute. Voting turns sampling into a confidence score. Search and tools take over where the model's own arithmetic fails. Each step up costs more tokens, so measure before paying for it.

**Done:** 2026-10-29 · Quiz 3/3 · Route row 2 of 109
**Lesson:** [phases/11-llm-engineering/02-few-shot-cot](../phases/11-llm-engineering/02-few-shot-cot/)
**My code:**
- [`labs/11-02-few-shot-cot/my_reasoning.py`](../labs/11-02-few-shot-cot/my_reasoning.py): `extract_answer`, `build_messages` (4 modes), `majority_vote`, `self_consistency`. 20/20 checks.
- [`experiment.py`](../labs/11-02-few-shot-cot/experiment.py): 12 multi-step word problems × 5 techniques on llama3.2 3B, measuring accuracy, output tokens and wall time.

**Run it:** `python3 labs/11-02-few-shot-cot/my_reasoning.py`, then `python3 labs/11-02-few-shot-cot/experiment.py` (needs `ollama serve`, ~6 min)

---

## Remember these 3 things

1. **The reasoning must come before the answer.**
   - Every generated token is another forward pass, and later tokens attend to it. Reasoning tokens are where intermediate results get stored.
   - "Answer first, then explain" gets no benefit. Under the causal mask, the answer token can't see reasoning that hasn't been written yet. The explanation is a justification written afterwards.
   - On my 3B model: **0/12 answer-only, 10/12 with CoT.**
2. **Few-shot examples teach format and style, not computation.**
   - Examples with answers only (no reasoning) scored **0/12**, the same as zero-shot.
   - Examples *with* reasoning matched zero-shot CoT's accuracy at **21% fewer tokens** (145 vs 184), because they showed how concise to be.
   - Use 3 to 5 examples that are similar to the input, cover the different cases, and use the exact output format. Never take them from the test set.
3. **Self-consistency costs N× and works best as a confidence signal.**
   - Sample N paths at **T > 0** and take a majority vote. At T=0 every sample is identical, so you get a fake 100% confidence and the escalation never triggers.
   - Mine cost **5.0×** the tokens for **+1** problem (92% vs 83%).
   - It **can't fix systematic errors.** The fuel problem got 5 different wrong answers. Its **confidence of 0.2 is the signal to route it** to a tool such as a calculator in a ReAct loop.

---

## In plain English

Ask someone to multiply 37 × 48 in their head and say only the result, and they'll often get it wrong. Give them paper and they'll get it right. Their knowledge didn't change; they got somewhere to put the intermediate numbers. That's chain-of-thought.

Few-shot examples are like showing a new hire three finished reports. They copy the layout and length well. If the reports only show the final number, though, they learn nothing about how to work it out.

Self-consistency is asking five people independently and going with the majority. If they all agree, you can be confident. If all five give different answers, the majority is meaningless, but you've learned something valuable: this question needs a calculator, not another opinion.

---

## Worked example: my results (llama3.2 3B, 12 problems)

```text
technique              accuracy   tokens/question   time
zero_shot              0/12   0%         4             3s   "Answer with ONLY the final number"
few_shot (answers)     0/12   0%         8             4s   examples without reasoning: format only
zero_shot_cot         10/12  83%       184            50s   "think step by step" -> +83 points for ~45x tokens
few_shot_cot          10/12  83%       145            41s   same accuracy, 21% fewer tokens
self_consistency n=5  11/12  92%       731           217s   5.0x few_shot_cot for +1
```

The two few_shot_cot misses are different kinds of failure:

```text
widgets (gold 18):   "...432 ÷ 24 = 18 ... The factory packs 18 full boxes each day."  -> extracted None
                     right maths, no "The answer is" line = FORMAT failure. Fix: enforce the format (11/03) or retry.
fuel (gold 39.78):   340 × 0.065 = 22.1 L; 22.1 × 1.80 = 39.78. The model got 40.18, 4, 40.26, 40.04...
                     REASONING failure that repeats on every path. SC votes split, conf 0.2. Fix: a calculator tool.
```

---

## Formulas / cheat sheet

| Thing | Formula / rule |
|-------|----------------|
| Majority vote confidence | `count(winner) / N`, with unparseable replies counted in N |
| Self-consistency cost | about N × the single CoT cost (tokens and latency, unless run in parallel) |
| ToT cost | up to `b + b² + … + b^d` calls. Branching 3, depth 3 is up to 39 calls |
| Escalation rule | single CoT (T=0) agrees with the SC majority AND conf ≥ 0.8 → accept; otherwise escalate (ToT or a tool) |
| Answer extraction | LAST "the answer is N". Normalise `$`, commas, trailing zeros. **No** last-number fallback |

---

## My code (with line notes)

```python
def extract_answer(text):
    text = text.strip()
    match = re.findall(r'answer is\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        answer = match[-1]                          # LAST one: models correct themselves mid-reply
    else:
        answer = text                               # answer-only reply: accept only a bare number
        if not re.fullmatch(r'-?\$?[\d,]+(?:\.\d+)?', answer):
            return None                             # no last-number fallback, which would grade intermediate values
    answer = answer.replace(",", "")
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".")     # "18.00" -> "18", "1327.50" -> "1327.5"
    return answer

def majority_vote(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return (None, 0.0)
    counts = Counter(valid)                          # most_common keeps first-seen order on ties (Python 3.7+)
    return (counts.most_common(1)[0][0], counts.most_common(1)[0][1] / len(answers))   # Nones in the denominator
```

**Fixes still pending:**
- `llm(messages, temperature, i)` is positional. Use `llm(messages, temperature=temperature, seed=i)`, because a wrapper with a different parameter order would silently swap them.
- The bare-number path allows `$` but never strips it (`"$42"` → `"$42"`). The "answer is" path has no minus sign (`"The answer is -5"` → None).
- `most_common(1)` is called twice. Unpack it once.

---

## Where this shows up in AI engineering

- **SEC RAG (Project 1):** "Did margin rise YoY?" is a multi-step question, so ask for the reasoning before the answer, and **compute the numbers with code, not the LLM.** The fuel problem showed that a 3B model can't multiply decimals reliably.
- **SRE agent (Project 2):** ReAct (thought → tool → observation) is the agent loop. Low self-consistency confidence is a reasonable trigger for "escalate to a human".
- **Cost control:** CoT added about 45× the output tokens here. Use it for multi-step tasks, and skip it for classification or lookups. Reasoning models (o-series, R1) already do CoT internally, so don't add "think step by step" for them.
- **Evals:** separate *format* failures from *reasoning* failures. They need different fixes (structured output vs better reasoning or tools).

---

## Mistakes to avoid (ones I made or nearly made)

- **Underestimating the CoT gap on small models.** I predicted 3/12 → 8/12. It was 0/12 → 10/12. Without scratch paper, a 3B model basically can't do multi-step arithmetic.
- **Thinking voting fixes everything.** Five wrong answers out-vote nothing. Self-consistency only helps when errors are *random*, not systematic.
- **Running self-consistency at T=0.** That gives 5 identical replies, 100% confidence, and an escalation that never fires, even when the answer is wrong.
- **Calling an extraction None a "wrong answer".** The widgets reply had the right maths. Always look at the raw reply before blaming the reasoning.

---

## Words

- **Few-shot:** input→output demonstrations in the prompt.
- **Chain-of-thought (CoT):** intermediate reasoning tokens before the final answer. Zero-shot ("think step by step") or few-shot (examples with reasoning).
- **Self-consistency:** sample N CoT paths at T > 0 and take a majority vote on the final answer.
- **Tree-of-thought (ToT):** generate candidate steps, have the LLM score them, prune, and expand the best (BFS/DFS).
- **ReAct:** Thought → Action (tool) → Observation loop; the basis of agents.
- **Prompt chaining:** sequential prompts where each output feeds the next input.

## Connects to

- [07/07 GPT](07-07-gpt.md): one forward pass per generated token, which is why reasoning tokens add computation.
- [07/02 Self-attention](07-02-self-attention.md): the causal mask is why answer-first reasoning can't help.
- [11/01 Prompt engineering](11-01-prompt-engineering.md): format instructions and the T=0-is-greedy weak spot.
- [01/06 Probability](01-06-probability-and-distributions.md): temperature > 0 is what makes the voting paths differ.

---

## Self-test

1. **"Give the answer on line 1, then explain." Is that CoT?**
   <details><summary>Answer</summary>No. The answer token is generated before any reasoning exists, and the causal mask means it can't attend to later tokens. The explanation is written afterwards and gives no accuracy benefit.</details>
2. **Few-shot examples with answers only scored 0/12, and with reasoning 10/12. What do examples teach?**
   <details><summary>Answer</summary>The pattern they demonstrate. Answer-only examples teach the output format; examples with reasoning teach the reasoning steps and their length (21% fewer tokens than zero-shot CoT).</details>
3. **Self-consistency gives 5 different answers on one problem. What do you do?**
   <details><summary>Answer</summary>Don't trust the "winner" (confidence 0.2). Escalate: route it to a tool (a calculator or code) or to a human. Voting can't fix an error the model makes on every path.</details>
4. **Why is self-consistency at T=0 worse than useless?**
   <details><summary>Answer</summary>Greedy decoding makes every sample identical, so confidence is always 100%. You pay N× and the "low confidence → escalate" rule never fires, even on wrong answers.</details>
5. **When is ToT worth 10 to 40× the cost?**
   <details><summary>Answer</summary>When the search space is wide and partial solutions can be evaluated: puzzles, planning, constrained generation. Not for arithmetic errors (use a tool) or simple lookups.</details>
6. **Your reply says "432 ÷ 24 = 18. The factory packs 18 boxes." and the extractor returns None. Is the model wrong?**
   <details><summary>Answer</summary>No. It's a format failure, not a reasoning failure. Enforce the answer format (structured output, or a retry) instead of adding a "last number" fallback, which would mis-grade other replies.</details>
