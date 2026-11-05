# 11/03 Structured Outputs: JSON, Schema Validation, Constrained Decoding

> Format is a decoding problem, not a prompting problem. A schema compiled into a token mask makes the format impossible to get wrong. Validation and retries are the fallback. Nothing in this stack checks whether the values are true.

**Done:** 2026-10-30 · Quiz 3/3 · Route row 3 of 109
**Lesson:** [phases/11-llm-engineering/03-structured-outputs](../phases/11-llm-engineering/03-structured-outputs/)
**My code:**
- [`labs/11-03-structured-outputs/my_structured.py`](../labs/11-03-structured-outputs/my_structured.py): `validate` (JSON Schema subset with `$.a[1].b` paths), `extract_with_retry` (the Instructor pattern), `constrained_enum_decode` (token masking). 20/20 checks.
- [`experiment.py`](../labs/11-03-structured-outputs/experiment.py): my `Filing` pydantic model (10/10 agreement with my validator), then prompt-only vs retry vs JSON mode vs schema mode on 6 SEC snippets with llama3.2.

**Run it:** `python3 labs/11-03-structured-outputs/my_structured.py`, then `python labs/11-03-structured-outputs/experiment.py` (needs pydantic and `ollama serve`)

---

## Remember these 3 things

1. **There are four levels of control, and only the decoder gives a guarantee.**
   - The levels: prompt ("return JSON") < JSON mode (always parses) < schema mode (exact keys, types, enums, required) < constrained decoding (how the last two are enforced).
   - Constrained decoding sets the logits of illegal tokens to −∞ before sampling. It's the same trick as top-k in 07/07, but the mask comes from a grammar compiled from the schema.
   - Mine: prompt-only **0/6** valid, JSON mode and schema mode **6/6** in one call each.
2. **Make fields required AND nullable, or the decoder makes up values.**
   - With `fiscal_year: int` (non-null) on a Tesla 8-K that states no year, the decoder **can't** emit null or skip the key. It produced `fiscal_year: 0`, which passes and gets stored as a confident wrong value.
   - Use `["integer", "null"]` (pydantic: `int | None` with no default). "Required" makes the model address the field; "nullable" lets it answer honestly.
   - Put a `"reasoning"` field **first** if the model needs to think. Keys are generated in schema order, so that's CoT inside the JSON.
3. **A schema checks the shape, never the truth. Retries are a fallback with their own risks.**
   - `"company": "HACKED"` (the injection) passed **every** mode. `period_end: "2024-06-03"` (the date of report, not a period end) passed too. Only prompts and an eval set catch these.
   - Retry feedback has to be actionable. A raw `char 0` error gave **0/6 in 18 calls**; plain words plus "reply with ONLY the JSON" gave **6/6 in 12**.
   - Even then, the retry **lost Apple's correct year and date**. Each retry is a fresh generation.

---

## In plain English

Asking a model for JSON is like asking someone to fill in a paper form by writing on a blank page. Usually they copy the layout; sometimes they add a cover note or forget a box. JSON mode is a page that must be a valid form of *some* kind. Schema mode is a printed form with labelled boxes, tick-boxes for enums, and digit-only boxes for numbers. You physically can't write letters in the year box.

There's a catch, though. If the year box can't be left blank and the person doesn't know the year, they'll write *something*, like 0. Always give the box a "not stated" option. And no form stops someone writing a convincing lie, such as "HACKED", in the name box.

---

## Worked example: my results (llama3.2 3B, 6 SEC snippets, T=0)

```text
strategy                  valid   correct   calls   failure
prompt only               0/6     0/6        6      "Here is the extracted metadata in JSON format:\n\n```..."
prompt + retry (v1)       0/6     0/6       18      raw "char 0" feedback -> model rewrote it as "Python code"
prompt + retry (v2)       6/6     4/6       12      fixed the format, LOST Apple's year/date; HACKED
JSON mode                 6/6     5/6        6      HACKED
schema (pydantic)         6/6     5/6        6      HACKED

Tesla 8-K, fiscal_year: int (not nullable), schema-constrained:
  {"company": "Tesla, Inc.", "form": "8-K", "fiscal_year": 0, "period_end": "2024-06-03"}   <- 0 forced; date invented
```

Constrained decoding in miniature (my TODO 3): the "model" wants `available` (logit 9), but legal tokens must keep the prefix inside `{in_stock, out_of_stock, preorder}`. The mask leaves `out` (2.0) as the best legal first token, then `_`, `of`, `_`, `stock`, and `<eos>` only once the prefix is complete. The result is `out_of_stock`.

---

## Formulas / cheat sheet

| Thing | What to write |
|-------|---------------|
| Ollama schema mode | `"format": Filing.model_json_schema()` (top level of the body) |
| Required + nullable (schema) | `"type": ["integer", "null"]` and list it in `"required"` |
| Required + nullable (pydantic) | `fiscal_year: int \| None = Field(ge=1990, le=2030)` (no default means required) |
| Enum | `form: Literal["10-K", "10-Q", "8-K"]` |
| No coercion, no extra keys | `model_config = ConfigDict(strict=True, extra="forbid")` |
| Date pattern | `Field(pattern=r"^\d{4}-\d{2}-\d{2}$")`; in Python, `re.fullmatch`, never `re.match` |
| Constrained-decoding legality | token t is legal iff `prefix + t` is a prefix of some allowed string; `<eos>` iff prefix ∈ allowed |

---

## My code (with line notes)

```python
class Filing(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")   # "2023" stays a string -> rejected; "cik" -> rejected
    company: str
    form: Literal["10-K", "10-Q", "8-K"]
    fiscal_year: int | None = Field(ge=1990, le=2030)       # becomes anyOf[integer(1990..2030), null] for the decoder
    period_end: str | None = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

def extract_with_retry(messages, schema, llm=call_llm, max_attempts=3, fmt=None):
    for attempt in range(1, max_attempts + 1):
        reply = llm(messages, fmt=fmt)
        ...                                                   # json.loads, then validate
        messages = messages + [                               # new list, so the caller's list is untouched
            {"role": "assistant", "content": reply},
            {"role": "user", "content": <plain-English errors> + "Reply with ONLY the JSON object."},
        ]

def constrained_enum_decode(next_logits, vocab, allowed):
    ...
    masked = [l if tok in legal else float("-inf") for tok, l in zip(vocab, logits)]   # make `legal` a set: O(V) not O(V²)
    next_token = vocab[masked.index(max(masked))]
```

**Bugs I fixed after the first 20/20:**
- `re.match` → `re.fullmatch`. `"2023-09-30 (restated)"` had been passing the pattern.
- The keyword checks were nested under `if "type" in schema:`, so `{"required": [...]}` and `{"enum": [...]}` without a `type` checked nothing. Pydantic's `anyOf` branches look like this.
- After a type mismatch, I now return instead of also reporting enum errors (noise that gets sent back to the model).
- Added `minItems`.

---

## Where this shows up in AI engineering

- **SEC RAG (Project 1):** the answer object `{reasoning, answer, citations[{id, quote}], confidence}` should be a pydantic model passed as the schema. The same model serves as the FastAPI response type.
- **SRE agent (Project 2):** tool calls are schema-constrained JSON arguments. Anthropic tool use takes `input_schema`, and OpenAI takes `json_schema`. Use enums for `severity` and `action`, so the model can't invent `"kinda_bad"`.
- **Evals:** schema-valid rate is just the first gate. Field-level accuracy against gold labels is what catches HACKED, the year 0, and a period_end that's really the report date.
- **Provider fallback:** where there's no schema mode (some local or older APIs), use validate + actionable retry. Log the number of attempts as a cost metric.

---

## Mistakes to avoid (ones I made or nearly made)

- **"A required non-null field will make the 8-K request fail."** The reverse is true under constrained decoding: it **succeeds with a made-up value** (`0`). Visible failures are better than silent ones.
- **Raw parser errors as retry feedback.** `Expecting value: line 1 column 1 (char 0)` led the model to debug "Python code". 18 calls, 0 fixes.
- **An empty pydantic model.** `class Filing(BaseModel): pass` accepted all 10 JSON strings, including garbage, and the decoder schema had no fields. A schema that declares nothing validates everything.
- **`re.match` for patterns.** It anchors only at the start. Use `fullmatch`.
- **Predicting 2/6 for prompt-only.** Without an explicit "no fences or preamble", llama3.2 wrapped all 6. That one line was worth 67 points in 11/01.

---

## Words

- **JSON mode:** the output is guaranteed to parse; no schema is enforced.
- **Schema mode / structured outputs:** the output matches a JSON Schema (keys, types, enums, required).
- **Constrained decoding / token masking:** set the logits of illegal tokens to −∞ at each step, using a grammar (FSM/PDA) compiled from the schema.
- **JSON Schema:** declarative contract: `type`, `properties`, `required`, `enum`, `pattern`, `minimum`, `items`, `additionalProperties`, `anyOf`.
- **Pydantic:** Python models → JSON Schema plus validation. `strict=True` turns off coercion.
- **Tool use / function calling:** the model emits a call with schema-typed arguments; the same mechanism.
- **Instructor pattern:** validate, then send the errors back, then retry.

## Connects to

- [07/07 GPT](07-07-gpt.md): logit masking (top-k) is the same mechanism constrained decoding uses.
- [11/01 Prompt engineering](11-01-prompt-engineering.md): the missing `}` that JSON mode fixed, and "no fences" being worth 67 points.
- [11/02 Few-shot / CoT](11-02-few-shot-cot.md): a `reasoning` field first equals CoT inside the schema; the missing answer line was a format failure.

---

## Self-test

1. **Your schema says `fiscal_year: integer`, required, and you enable schema-constrained decoding on an 8-K with no year. What happens?**
   <details><summary>Answer</summary>The decoder can only produce digits, so the model invents an integer (it gave 0). The output validates and gets stored as wrong data. Make the field required and nullable.</details>
2. **JSON mode and schema mode both scored 6/6 valid and 5/6 correct. When would schema mode beat JSON mode?**
   <details><summary>Answer</summary>When the prompt doesn't pin down the keys and types. The vague prompt in 11/01 with JSON mode was 100% valid but had 0% right keys. Schema mode enforces names, types, enums and required fields regardless of the prompt.</details>
3. **Why did the first retry loop get 0/6 after 18 calls?**
   <details><summary>Answer</summary>The feedback was a raw parser error ("char 0") plus "correct the errors". The model misread it as a bug in the user's input and wrote Python. Feedback has to say what was wrong in plain words and ask for only the JSON.</details>
4. **Name two errors that pass any schema.**
   <details><summary>Answer</summary>Injected or hallucinated values (`"company": "HACKED"`), and plausible-but-wrong values (`period_end` set to the report date). The types are right, so only field-level evals or human review catch them.</details>
5. **What does `ConfigDict(strict=True, extra="forbid")` change?**
   <details><summary>Answer</summary>strict: no coercion, so "2023" is not accepted as 2023 and true is not 1. forbid: unknown keys such as "cik" are rejected instead of silently dropped.</details>
6. **In constrained decoding over `{in_stock, out_of_stock, preorder}`, when is `<eos>` legal after the prefix "pre"?**
   <details><summary>Answer</summary>Never. `<eos>` is legal only when the prefix is exactly an allowed value. After "pre", only "order" is legal, which gives "preorder".</details>
