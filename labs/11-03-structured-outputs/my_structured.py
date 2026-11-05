# Lesson 11/03 Structured Outputs: a JSON Schema validator, validate-and-retry, and enum-constrained decoding.
# Run: python3 labs/11-03-structured-outputs/my_structured.py     (stdlib only; self-checks need no model)
# The live experiment (experiment.py) calls llama3.2 through Ollama and compares against pydantic.
import json
import re
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Provided: one chat call through Ollama. `fmt` is None (free text), "json" (JSON mode),
# or a JSON Schema dict (schema-constrained decoding).
# ---------------------------------------------------------------------------

def call_llm(messages, temperature=0.0, seed=0, fmt=None, max_tokens=300):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens}}
    if fmt is not None:
        body["format"] = fmt
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

JSON_TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")


def validate(data, schema, path="$"):
    errors = []

    # 1. Type validation
    if "type" in schema:
        expected_types = schema["type"]

        if isinstance(expected_types, str):
            expected_types = [expected_types]

        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "null": type(None)
        }

        if not any(
            isinstance(data, type_map[t])
            and not (t in ("integer", "number") and isinstance(data, bool))
            for t in expected_types
        ):
            errors.append(
                f"{path}: expected {'|'.join(expected_types)}, "
                f"got {type(data).__name__}"
            )
            return errors  # Stop on type mismatch

    # 2. Enum validation
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: value {data} not in enum {schema['enum']}")

    # 3. Number validation
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")

        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    # 4. String pattern validation
    if isinstance(data, str) and "pattern" in schema:
        if not re.fullmatch(schema["pattern"], data):
            errors.append(f"{path}: does not match pattern")

    # 5. Object validation
    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required")

        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                errors.extend(
                    validate(data[key], subschema, f"{path}.{key}")
                )

        if schema.get("additionalProperties") is False:
            allowed_keys = set(schema.get("properties", {}))

            for key in data:
                if key not in allowed_keys:
                    errors.append(f"{path}.{key}: not allowed")

    # 6. Array validation
    if isinstance(data, list):
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")

        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"{path}: exceeds maxItems")

        if "items" in schema:
            for i, item in enumerate(data):
                errors.extend(
                    validate(item, schema["items"], f"{path}[{i}]")
                )

    return errors   




def extract_with_retry(messages, schema, llm=call_llm, max_attempts=3, fmt=None):
    # TODO 2: the Instructor pattern, without the library.
    #   Each attempt: reply = llm(conversation, fmt=fmt); try json.loads; then validate(parsed, schema).
    #   - Parse error -> errors = ["$: invalid JSON: <the exception message>"]
    #   - Valid -> return {"data": parsed, "attempts": n, "errors": []}
    #   - Otherwise, append the model's reply as an "assistant" message and a "user" message that lists
    #     the errors (one per line) and asks for corrected JSON only. Then try again.
    #   Don't mutate the caller's `messages` list.
    #   After max_attempts failures -> {"data": None, "attempts": max_attempts, "errors": <last errors>}
    for attempt in range(1, max_attempts + 1):
        reply = llm(messages, fmt=fmt)
        try:
            parsed = json.loads(reply)
            errors = validate(parsed, schema)
            if not errors:
                return {"data": parsed, "attempts": attempt, "errors": []}
        except json.JSONDecodeError as e:
            errors = [f"$: invalid JSON: {str(e)}"]

        # Append the model's reply and the error message to the conversation
        messages = messages + [
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON or did not match "
                    "the required schema.\n"
                    "Fix the following errors:\n"
                    + "\n".join(errors)
                    + "\n\nReturn ONLY the corrected JSON object. "
                    "Do not include Markdown code fences, explanations, "
                    "preambles, or any text outside the JSON."
                )
            }
        ]
    return {"data": None, "attempts": max_attempts, "errors": errors}


def constrained_enum_decode(next_logits, vocab, allowed):
    prefix = ""

    while True:
        logits = next_logits(prefix)
        legal_tokens = []

        for token in vocab:
            if token == "<eos>":
                if prefix in allowed:
                    legal_tokens.append(token)
            else:
                candidate = prefix + token

                if any(a.startswith(candidate) for a in allowed):
                    legal_tokens.append(token)

        if not legal_tokens:
            raise ValueError("No legal tokens available")

        masked_logits = [
            logit if token in legal_tokens else float("-inf")
            for token, logit in zip(vocab, logits)
        ]

        next_token = vocab[masked_logits.index(max(masked_logits))]

        if next_token == "<eos>":
            break

        prefix += next_token

    return prefix  # Outside the while loop!

# ---------------------------------------------------------------------------
# Self-checks (offline). Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


FILING_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "form": {"type": "string", "enum": ["10-K", "10-Q", "8-K"]},
        "fiscal_year": {"type": ["integer", "null"], "minimum": 1990, "maximum": 2030},
        "period_end": {"type": ["string", "null"], "pattern": r"\d{4}-\d{2}-\d{2}"},
    },
    "required": ["company", "form", "fiscal_year", "period_end"],
    "additionalProperties": False,
}

if __name__ == "__main__":
    ok = []
    good = {"company": "Apple Inc.", "form": "10-K", "fiscal_year": 2023, "period_end": "2023-09-30"}
    ok.append(check("1 valid object", validate(good, FILING_SCHEMA) == [], str(validate(good, FILING_SCHEMA))))
    ok.append(check("1 nulls allowed", validate({**good, "fiscal_year": None, "period_end": None}, FILING_SCHEMA) == []))
    e = validate({**good, "fiscal_year": "2023"}, FILING_SCHEMA)
    ok.append(check("1 string year rejected", len(e) == 1 and e[0].startswith("$.fiscal_year"), str(e)))
    e = validate({**good, "fiscal_year": True}, FILING_SCHEMA)
    ok.append(check("1 bool is not an integer", len(e) == 1 and e[0].startswith("$.fiscal_year"), str(e)))
    e = validate({**good, "fiscal_year": 2023.0}, {"type": "integer"})
    ok.append(check("1 float is not an integer", len(e) == 1, str(e)))
    e = validate({**good, "form": "10K"}, FILING_SCHEMA)
    ok.append(check("1 enum", len(e) == 1 and e[0].startswith("$.form"), str(e)))
    e = validate({**good, "fiscal_year": 1850}, FILING_SCHEMA)
    ok.append(check("1 minimum", len(e) == 1 and e[0].startswith("$.fiscal_year"), str(e)))
    e = validate({**good, "period_end": "Sept 30, 2023"}, FILING_SCHEMA)
    ok.append(check("1 pattern", len(e) == 1 and e[0].startswith("$.period_end"), str(e)))
    e = validate({"company": "X", "form": "8-K", "note": "hi"}, FILING_SCHEMA)
    ok.append(check("1 missing + extra keys", sorted(x.split(":")[0] for x in e) == ["$.fiscal_year", "$.note", "$.period_end"], str(e)))
    ok.append(check("1 not an object", len(validate([good], FILING_SCHEMA)) == 1))
    arr = {"type": "array", "items": {"type": "number", "minimum": 0}, "maxItems": 2}
    e = validate([1, -2, 3], arr)
    ok.append(check("1 array items + maxItems", any(x.startswith("$[1]") for x in e) and len(e) == 2, str(e)))
    nested = {"type": "object", "properties": {"cites": {"type": "array", "items": {"type": "object", "required": ["id"]}}}}
    e = validate({"cites": [{"id": 1}, {}]}, nested)
    ok.append(check("1 nested path", e == ["$.cites[1].id: required"] or (len(e) == 1 and e[0].startswith("$.cites[1].id")), str(e)))

    replies = iter(['Sure: {"company": "Apple Inc."}',
                    '{"company": "Apple Inc.", "form": "10K", "fiscal_year": 2023, "period_end": "2023-09-30"}',
                    json.dumps(good)])
    convs = []
    def fake_llm(messages, temperature=0.0, seed=0, fmt=None, max_tokens=300):
        convs.append([dict(m) for m in messages])
        return next(replies)
    start = [{"role": "user", "content": "extract"}]
    r = extract_with_retry(start, FILING_SCHEMA, llm=fake_llm)
    ok.append(check("2 succeeds on attempt 3", r["data"] == good and r["attempts"] == 3 and r["errors"] == [], str(r)))
    ok.append(check("2 feedback grows the conversation", [len(c) for c in convs] == [1, 3, 5], str([len(c) for c in convs])))
    ok.append(check("2 errors sent back", convs[1][1]["role"] == "assistant" and convs[1][2]["role"] == "user"
                    and "invalid JSON" in convs[1][2]["content"] and "$.form" in convs[2][4]["content"], str(convs[-1][-1])))
    ok.append(check("2 caller's list untouched", start == [{"role": "user", "content": "extract"}]))
    r = extract_with_retry(start, FILING_SCHEMA, llm=lambda m, **k: '{"company": 5}', max_attempts=2)
    ok.append(check("2 gives up", r["data"] is None and r["attempts"] == 2 and len(r["errors"]) >= 1, str(r)))

    vocab = ["in", "_", "stock", "out", "pre", "order", "available", "<eos>"]
    allowed = ["in_stock", "out_of_stock", "preorder"]
    vocab2 = vocab + ["of"]
    def wants_available2(prefix):
        # the "model" loves "available" and would happily stop anywhere
        prefs = {"available": 9.0, "<eos>": 5.0, "out": 2.0, "in": 1.0, "_": 0.5, "stock": 0.4, "pre": 0.3, "order": 0.2, "of": 0.1}
        return [prefs[t] for t in vocab2]
    got = constrained_enum_decode(wants_available2, vocab2, allowed)
    ok.append(check("3 forced into the enum", got in allowed, repr(got)))
    ok.append(check("3 picks the best LEGAL path", got == "out_of_stock", repr(got)))
    def wants_pre(prefix):
        prefs = {"pre": 3.0, "order": 2.5, "<eos>": 2.8, "available": 1.0, "in": 0.1, "out": 0.1, "_": 0.1, "stock": 0.1, "of": 0.0}
        return [prefs[t] for t in vocab2]
    got = constrained_enum_decode(wants_pre, vocab2, allowed)
    ok.append(check("3 no early <eos> on 'pre'", got == "preorder", repr(got)))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
