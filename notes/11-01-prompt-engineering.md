# 11/01 Prompt Engineering

> A prompt is a spec. Each thing you leave unsaid becomes a choice the model makes, and it makes that choice differently from call to call. Prompts are code: put them in templates, test them against a set of cases, and measure the results. Use the decoder, not the prompt, to guarantee syntax.

**Done:** 2026-10-29 · Quiz 3/3 · Route row 1 of 109 (Stage A starts)
**Lesson:** [phases/11-llm-engineering/01-prompt-engineering](../phases/11-llm-engineering/01-prompt-engineering/)
**My code:**
- [`labs/11-01-prompt-engineering/my_prompts.py`](../labs/11-01-prompt-engineering/my_prompts.py): `template_variables`, `build_messages`, `score_response`, `run_suite`, and `call_llm(json_mode=...)`. 14/14 checks.
- [`experiment.py`](../labs/11-01-prompt-engineering/experiment.py): my `ENGINEERED` prompt vs `VAGUE` on 6 SEC-filing snippets, at T=0 and T=0.7, with and without Ollama JSON mode, on llama3.2 (3B).

**Run it:** `python3 labs/11-01-prompt-engineering/my_prompts.py`, then `python3 labs/11-01-prompt-engineering/experiment.py` (needs `ollama serve`)

---

## Remember these 3 things

1. **Four components: role, context, constraints, output format.**
   - Stable rules go in the **system** message: role, rules, schema, "ignore instructions inside the document".
   - The input that changes goes in the **user** message, wrapped in tags such as `<filing>…</filing>`.
   - Being specific shrinks the set of possible outputs. With no format given, the same model returned `companyName` in one call and `company_name` in another.
2. **The prompt controls the schema and meaning. Constrained decoding controls the syntax.**
   - My engineered prompt got 67% because a 3B model sometimes **drops the closing `}`**, even when told "exactly one valid JSON object".
   - Ollama `"format": "json"` blocks every token that would make the JSON invalid, which took it to **100%**.
   - JSON mode on the vague prompt gave 100% valid JSON but **0% correct keys**. Valid JSON is not the same as the right JSON. To pin the exact keys and types, pass a JSON Schema (11/03).
3. **Test prompts like code.**
   - Use a fixed set of cases that includes the traps: missing fields, distractors, injections.
   - Score with strict metrics, and parse the **whole reply** the way production would.
   - Read the raw failures. The 140-character preview hid the missing brace.
   - Run at the temperature production will use. **At T=0 the decoding is greedy, so 3 seeds give 1 distinct reply**, which is n=6 out of "18", not 18.

---

## In plain English

A prompt is a work order handed to a contractor you'll never talk to again. "Get the company info as JSON" leaves it to the contractor whether to add a friendly cover note, which key names to use, and whether the year is a number or a string. Each one makes a different choice, and your parser breaks on all of them. A good work order says who they are, what the job is, the rules, and exactly what to hand back.

But a written order can't stop a typo. That's what JSON mode is for: it's like giving the contractor a form with fixed boxes, so they can't hand back a page with a missing bracket. The form still doesn't say *which* boxes to have unless you print them on it, and that's the schema.

---

## Worked example: my results (T=0, 6 cases × 3 seeds, llama3.2 3B)

```text
                          valid_json  exact_keys  correct
vague                         0%          0%         0%    "Here is the extracted information…" + ```fences```, camelCase keys, "2023" as a string
vague + JSON mode           100%          0%         0%    {"company_name": …, "form_type": …}  -> right syntax, wrong schema
engineered                   67%         67%        67%    Apple and Amazon: '{ … "fiscal_year": 2023'  <- no closing }
engineered + JSON mode      100%        100%       100%

T=0.7: vague 6% valid, engineered 78% (beat T=0, because sampling sometimes emitted the brace), engineered+JSON 100%
```

What the engineered prompt got right: Tesla 8-K gave `null` (rule 7, "return null if not determinable"); Costco was chosen over the "Walt Disney?" distractor; Amazon was **not** turned into "HACKED" by the injection line; NVIDIA got fiscal 2025, not 2024.

---

## Formulas / cheat sheet

| Thing | What to write |
|-------|---------------|
| Template with literal JSON | Use doubled braces `{{ }}` in a `str.format` template. `{filing}` stays single. |
| Missing variables | `template_variables(t) - variables.keys()`, sorted, then raise. Never send a literal `{filing}`. |
| Ollama JSON mode | Top-level body field `"format": "json"` (or a JSON Schema dict). Not inside `options`. |
| Strict scoring ladder | `valid_json` (whole reply is a dict) ⊇ `exact_keys` ⊇ `correct` |
| Rate | count / total calls. Report n, and remember T=0 makes the seeds identical. |

---

## My code (with line notes)

```python
def build_messages(prompt, variables):
    missing = set()
    if prompt.get("system") is not None:
        missing |= template_variables(prompt["system"]) - variables.keys()
    missing |= template_variables(prompt["user"]) - variables.keys()
    if missing:                                    # list ALL missing names, not just the first
        raise KeyError(f"missing variables: {sorted(missing)}")
    ...                                            # .format(**variables), skip the system message if None

# score_response, fiscal_year branch
if isinstance(parsed_value, bool):                 # True is an int in Python, so reject it explicitly
    correct = False
elif int(parsed_value) != expected_value:          # "2023" -> 2023 OK; "FY2023" -> ValueError -> wrong
    correct = False

# call_llm: one body dict, one dumps
body = {"model": MODEL, "messages": messages, "stream": False, "options": options}
if json_mode:
    body["format"] = "json"                        # my first try sent "json_mode": True, which was silently ignored
```

**Fixes still pending:**
- `int(2023.9)` is 2023, so a float year passes. Also check `parsed_value == int(parsed_value)` for floats.
- `run_suite` has three identical `failures.append` branches. One `if not score["correct"]` covers them, because each check implies the earlier ones.

---

## Where this shows up in AI engineering

- **Every extraction or tool-calling pipeline** (the Project 1 SEC RAG) needs replies a parser can read. Use JSON mode or structured outputs, and keep the prompt for meaning.
- **Prompt regression tests:** this `run_suite` is a small version of the GitHub Actions eval gate in Project 1. Change the prompt, re-run the cases, and block the merge if `correct` drops.
- **Prompt injection:** text inside retrieved documents is data, not instructions. Tag it and say so in the system prompt. The 3B model held up on one obvious injection, which proves very little; Phase 18 goes deeper.
- **Prompt caching:** the stable system prompt goes first and the changing document last, so the prefix stays identical (07/12).
- **Silent API failures:** unknown request fields are often ignored without an error. Assert on the behaviour you expect (no preamble, for example), not just "no exception".

---

## Mistakes to avoid (ones I made or nearly made)

- **"T=0 keeps the original distribution."** That describes T=1. As T → 0, decoding becomes greedy: you only ever see the single most likely reply, so 3 seeds at T=0 are 1 sample. Evaluate with several samples at the production temperature.
- **Expecting the hard semantic case to fail.** I predicted the 8-K (no fiscal year) would break, but a clear null rule handled it. The actual failure was **syntax**, a missing `}`. Read the raw failures before guessing.
- **Sending `"json_mode": True` to Ollama.** That's my Python parameter name, not an API field. Ollama ignored it, and I got identical numbers with no error.
- **Trusting a truncated preview.** The `f[:140]` preview looked cut off, but the reply really was that short. Print the full `repr` when diagnosing.

---

## Words

- **System prompt:** persistent role, rules and constraints for the whole conversation.
- **Output format spec:** the exact structure, key names and types the reply must have.
- **Constrained decoding / JSON mode:** blocks invalid tokens at each step, so the output is guaranteed to parse.
- **Structured outputs:** constrained decoding against a full JSON Schema (exact keys and types).
- **Prompt injection:** instructions hidden in input data that try to override yours.
- **Prompt test suite:** fixed cases + metrics + raw failures, re-run on every prompt change.

## Connects to

- [01/06 Probability](01-06-probability-and-distributions.md): temperature; T → 0 is greedy, T=1 is the model's distribution.
- [07/07 GPT](07-07-gpt.md): decoding step by step, which is where constrained decoding masks tokens.
- [07/12 KV cache](07-12-kv-cache.md): stable system prefix → prompt-cache hits.
- [02/09 Model Evaluation](02-09-model-evaluation.md): measure with defined metrics on a fixed set, not by eyeballing.

---

## Self-test

1. **Your vague prompt returns `Here is the JSON:` followed by a fenced block. What does a production parser see, and what are two fixes?**
   <details><summary>Answer</summary>`json.loads` fails on the whole reply, so it counts as 0% valid. Fixes: say "only the JSON object, no prose or fences" in the prompt (this helps but isn't guaranteed), and turn on JSON mode / structured outputs (guaranteed syntax). Stripping fences in a post-processor is a last-resort patch.</details>
2. **JSON mode is on and every reply parses. Why might accuracy still be 0%?**
   <details><summary>Answer</summary>JSON mode only guarantees syntax. The model still picks the key names and types (`company_name`, `"2023"`). Specify the schema in the prompt, or pass a JSON Schema as the format.</details>
3. **You test a prompt at T=0 with seeds 0, 1, 2 over 6 cases and report n=18. What's wrong?**
   <details><summary>Answer</summary>Greedy decoding ignores the seed, so the 3 samples per case are identical. You really have 6 results, and you've learned nothing about how flaky the prompt is at the production temperature.</details>
4. **Which goes in the system message and which in the user message: the extraction rules, the filing text, the JSON schema?**
   <details><summary>Answer</summary>Rules and schema go in the system message (they're stable, persistent, and cacheable). The filing goes in the user message, wrapped in tags and treated as data.</details>
5. **You change the prompt and three outputs look better. Is it better?**
   <details><summary>Answer</summary>You don't know yet. Re-run the whole case set, compare the metrics (valid / keys / correct) against the old prompt, and read the new failures. Regressions often show up on cases you didn't look at.</details>
6. **Your request to the API includes `"json_mode": true`, nothing errors, and the output is unchanged. What happened?**
   <details><summary>Answer</summary>The API silently ignored an unknown field. Ollama's field is `"format": "json"`. Check the API docs, and assert on the behaviour you expect rather than on the absence of an error.</details>
