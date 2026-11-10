# 11/09 Function Calling & Tool Use

> The model never runs anything. It writes JSON naming a function and its arguments; **my code** validates it, runs it, and sends the result back as a `tool` message, then loops. On llama3.2 (3B), 5 fictional companies, 10 questions: closed-book 0/10 (it refused), tools with a strict validator 3/10, with lenient type coercion 5/10. Growth questions stayed **0/5** because the 3B model can't chain calls. Vague schemas got 0/10 **with 3 confident wrong numbers**. It never retried after an error, and `max_iters` was the only thing that stopped a "call again" tool.

**Done:** 2026-11-10 · Quiz 3/3 · Route row 11 of 109
**Lesson:** [phases/11-llm-engineering/09-function-calling](../phases/11-llm-engineering/09-function-calling/)
**My code:**
- [`labs/11-09-function-calling/my_tools.py`](../labs/11-09-function-calling/my_tools.py): `validate_args` (JSON Schema subset: types, required, unknown args, enum, min/max), `ToolRegistry` (`schemas()`, `call()` returns error dicts and never raises), `run_loop` (model → tool_calls → results → repeat, capped at `max_iters`). 21/21 checks.
- [`experiment.py`](../labs/11-09-function-calling/experiment.py): my loop between llama3.2 (Ollama native tools) and a fictional financial DB: closed-book vs strict vs lenient, clear vs vague schemas, bare vs actionable errors, a never-ready tool.

**Run it:** `python labs/11-09-function-calling/my_tools.py`, then `python labs/11-09-function-calling/experiment.py` (needs `ollama serve` + llama3.2, ~2 min)

---

## Remember these 3 things

1. **The loop is the whole pattern, and the model only proposes.**
   - Send messages + tool schemas → the model returns `tool_calls` → validate → execute → append `{"role": "tool", ...}` → call the model again → stop when it answers with no tool calls, or at `max_iters`.
   - Everything between "the model asked" and "the tool ran" is **my code**: allowlist, validation, permissions, timeouts.
2. **Never raise inside the loop. Return a structured error.**
   - A raised exception kills the whole request: the user gets a 500 and **every earlier tool result is lost**.
   - The transcript also has a `tool_call` with no matching result, and OpenAI/Anthropic **reject the next request** when a `tool_call_id` is unanswered.
   - An error dict (`UNKNOWN_TOOL` / `INVALID_ARGS` / `TOOL_FAILED`) lets the model explain or retry.
3. **The schema is a prompt, and the model you ship decides what breaks.**
   - Vague (`tool_a`, "Data.", x/y/z): 0/10 and **3 wrong numbers**, worse than no tools (0 wrong).
   - The 3B model sent `"2024"` as a string (lenient coercion: 3 → 5/10), nested `get_financial` inside `pct_change` (growth 0/5), and retried **0 times** after an error, even when told how to fix it.
   - Test with the production model. A frontier model fails in different places.

---

## In plain English

Giving a model tools is like giving an intern a phone list. The intern can't make the calls; they write a sticky note saying "call Finance, ask for Halcyon's 2024 revenue", and you make the call and hand back the answer. Your job is to check the note makes sense (is that a real department, is the year a number), to never let a bad note crash the office, and to stop the intern after 8 notes if they keep asking for a report that's never ready. A small intern reads the phone list literally: if it says "tool_a: Data.", they'll guess, and then confidently tell the customer a number.

---

## Worked example

**Counting calls by hand:** "Weather in Tokyo and Paris, in °F?" with `get_weather` (°C) and `c_to_f`.
- Parallel: model call 1 returns *both* weather calls at once (independent), model call 2 returns both conversions, model call 3 answers = **3 model calls, 4 tool calls**.
- I answered "4 calls" (1 model, 1 round of weather, 1 conversion, 1 final model): I mixed rounds of tool calls with model calls. Count the two kinds separately: model calls are the latency and cost.
- Parallel only works for **independent** calls: the conversions need the weather results first.

**My experiment** (llama3.2 3B, temperature 0, `num_ctx` 4096; 5 fictional companies × FY2022–2024; tools `get_financial(company, fiscal_year: int, metric: enum)` and `pct_change(old, new)`):

| run | correct /10 | lookups /5 | growth /5 | wrong numbers stated | tool errors |
|---|---|---|---|---|---|
| closed-book | 0 | 0 | 0 | **0** (refused) | n/a |
| tools, strict validator | 3 | 3 | 0 | 0 | 7 |
| tools, lenient (coerce `"2024"`) | **5** | 5 | 0 | 0 | 5 |
| vague schemas (lenient) | **0** | 0 | 0 | **3** | 12 |

- **Growth trace:** `pct_change(old="get_financial", new="get_financial", company=..., fiscal_year="2024")`: it tried to do two lookups and a division in one call. My validator blocked all of them.
- **Error style (strict, 5 lookups):** bare `{"error": "INVALID_ARGS"}` 3/5, message + "call it again" 3/5. Retries after an error: **0/5 and 0/5**.
- **Never-ready tool, `max_iters` 8:** plain prompt → 2 model calls, then a **fabricated report**. "Call again until ready" prompt → 8 model calls → `STOPPED: hit max_iters`.
- 11 invalid calls caught; 101 model calls in 104 s.

**My predictions:**
- closed-book 0/10 ✓
- strict 6/10 ✗ (3: string years)
- lenient 10/10, growth 5/5 ✗ (5/10, growth 0/5: can't chain)
- vague 3/10 ✗ (0/10, and 3 wrong numbers)
- retries: not sure (0/5 both)
- +retry prompt: 8 model calls ✓

---

## Formulas / cheat sheet

- **Tool definition (OpenAI/Ollama shape):** `{"type": "function", "function": {"name", "description", "parameters": <JSON Schema>}}`. Anthropic uses `{"name", "description", "input_schema"}`.
- **Tool result message:** Ollama `{"role": "tool", "tool_name", "content"}`; OpenAI `{"role": "tool", "tool_call_id", "content"}`; Anthropic a `tool_result` block with `tool_use_id`. **Every call needs a result**, even an error.
- **Arguments:** OpenAI returns a JSON **string** (`json.loads`, can fail → `BAD_JSON`); Ollama returns a **dict**.
- **`tool_choice`:** `auto` (model decides) / `required` or `any` (must call one) / a specific function (force it, e.g. for extraction).
- **Error dict:** `{"error": CODE, "message": <what was wrong>, "next_step": <what to do>}`. Codes: `UNKNOWN_TOOL` (+ the available list), `INVALID_ARGS`, `TOOL_FAILED`, `BAD_JSON`.
- **Loop guards:** `max_iters`, a total wall-clock timeout, per-tool timeouts, and detecting the same call repeated.
- **Serialization:** `json.dumps(result, default=str)`, otherwise a `datetime`/`Decimal` result crashes the loop outside the try.

---

## My code (what I wrote and why)

```python
match expected:                                   # validate_args: one branch per JSON Schema type
    case "integer":
        ok = isinstance(v, int) and not isinstance(v, bool)   # True is an int in Python
    ...
if ok and "enum" in spec and v not in spec["enum"]:           # only check enum/bounds once the type is right
```

```python
if name not in self.tools:
    return {"error": "UNKNOWN_TOOL", "message": f"... available: {list(self.tools)}"}
errors = validate_args(args, schema)
if errors:
    return {"error": "INVALID_ARGS", "message": "; ".join(errors)}   # the tool never runs
try:
    return fn(**args)
except Exception as e:
    return {"error": "TOOL_FAILED", "message": f"{type(e).__name__}: {e}"}  # never raise
```

```python
for _ in range(max_iters):
    output = chat(messages, tools)
    messages.append(output)
    if not output.get("tool_calls"):
        return output["content"], trace
    for call in output["tool_calls"]:              # every call gets a tool message back
        ...
return "STOPPED: hit max_iters", trace
```

**Review notes to fix:** `case _:` rejects tools with `array`/`object` params (an unknown *schema* type is my gap, not the model's error); `json.dumps(result)` needs `default=str`; use `.get("content", "")`.

---

## Where this shows up in AI engineering

- **Project 2 (agents):** this loop is the core of every agent. Add an allowlist, argument validation, `max_iters` plus a timeout, and log the full trace (calls, args, results, latency).
- **Security:** the model's arguments are **untrusted input**. No model-written SQL or shell, parameterized queries only, least-privilege tools, a human confirmation step for anything destructive (send, delete, pay), and watch for prompt injection inside tool *results*.
- **Small vs frontier models:** a 3B can do single lookups but not dependent chains. Fixes: a combined tool (`get_growth`), plan in code, or a bigger model for the planner.
- **Interviews:** "Does the LLM run the function?", "Walk me through the loop", "How do you stop runaway loops?", "Raise or return?", "Parallel tool calls: when?", "How do you secure tool use?"

---

## Mistakes to avoid

- **Raising inside a tool:** it kills the request and leaves an orphan `tool_call_id`.
- **No iteration cap:** a "not ready" tool plus a persistent prompt = an endless loop (mine hit 8/8).
- **Vague names and descriptions:** worse than no tools, because the model guesses arguments and then states wrong numbers.
- **Over-strict types with no coercion layer:** `"2024"` vs `2024` cost 2 of 5 lookups. Decide deliberately: coerce safe cases, reject the rest.
- **Assuming an actionable error makes the model retry:** the 3B never did. Measure retries on your model.
- **Parallelizing dependent calls:** a conversion can't run before the lookup it needs.
- **Counting "calls" loosely:** model calls (latency, cost) and tool calls are different numbers.
- **Trusting tool arguments:** that's how SQL injection gets in.

---

## Words

- **Function calling / tool use:** the model emits a structured request to call a named function with JSON arguments.
- **Tool schema:** name + description + JSON Schema parameters; the model's only knowledge of the tool.
- **Tool loop / agent loop:** model → tool calls → results → model, until a final answer or a cap.
- **`tool_choice`:** auto / required / a specific tool.
- **Parallel tool calls:** several independent calls in one model turn.
- **Lenient (lax) validation:** coerce safe types (`"2024"` → 2024) before validating.
- **`max_iters`:** the hard cap on loop rounds.

## Connects to

- [11-03 Structured Outputs](11-03-structured-outputs.md): tool arguments are structured output; `validate_args` is my 11/03 `validate` again, and the retry-feedback lesson is the same (here the 3B didn't retry at all).
- [11-01 Prompt Engineering](11-01-prompt-engineering.md): descriptions are prompts; T=0 for tool calls.
- [11-05 Context Engineering](11-05-context-engineering.md): every tool result is appended to the context, so long loops eat the window (`num_ctx`).
- [07-07 GPT](07-07-gpt.md): a tool call is just generated tokens.
- Next: 11/10 Evaluation & Testing LLM Applications.

---

## Self-test

1. **Does the LLM execute the function?**
   <details><summary>Answer</summary>No. It generates JSON naming the function and arguments; my code validates and runs it, then sends the result back.</details>
2. **Walk through the loop and its stop conditions.**
   <details><summary>Answer</summary>Messages + schemas → model → if tool_calls: validate, execute, append a tool message per call, repeat; stop when the reply has no tool calls, or at max_iters (plus a timeout).</details>
3. **Why return an error dict instead of raising?**
   <details><summary>Answer</summary>Raising kills the request (500, earlier tool work lost) and leaves an unanswered tool_call_id the API rejects. An error dict lets the model explain or fix its arguments.</details>
4. **Strict validator 3/10, lenient 5/10. What was the difference?**
   <details><summary>Answer</summary>The model sent fiscal_year "2024" as a string on 2 lookups; coercing numeric strings before validation fixed both.</details>
5. **Why was growth 0/5 even with tools?**
   <details><summary>Answer</summary>It needs two dependent rounds (look up both years, then pct_change). The 3B tried it in one call, passing "get_financial" as the number. Fix: a combined tool, planning in code, or a bigger model.</details>
6. **Closed-book stated 0 wrong numbers; vague tools stated 3. Why is that worse?**
   <details><summary>Answer</summary>Closed-book refused. Vague tools got the model to guess arguments, hit errors, and then answer confidently anyway. Bad tools, like bad RAG context, breed confident wrong answers.</details>
7. **When can tool calls run in parallel?**
   <details><summary>Answer</summary>Only when independent: weather for Tokyo and Paris yes; the °F conversion must wait for the weather result.</details>
8. **A tool returns a datetime. What breaks in my loop?**
   <details><summary>Answer</summary>json.dumps(result) raises TypeError outside the try. Use json.dumps(result, default=str).</details>
