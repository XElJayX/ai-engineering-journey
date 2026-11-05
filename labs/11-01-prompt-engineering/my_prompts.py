# Lesson 11/01 Prompt Engineering: prompt templates, a response scorer and a test-suite runner.
# Run: python3 labs/11-01-prompt-engineering/my_prompts.py     (stdlib only; self-checks need no model)
# The live experiment (experiment.py) calls llama3.2 through Ollama at http://localhost:11434.
import json
import re
import string
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Provided: one LLM call through Ollama's chat API (stdlib HTTP, no SDK).
# ---------------------------------------------------------------------------

def call_llm(messages, temperature=0.0, seed=None, max_tokens=200,json_mode=False):
    # messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -> the reply text
    options = {"temperature": temperature, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = seed
    body = json.dumps({"model": MODEL, "messages": messages, "stream": False, "options": options}).encode()
    if json_mode:
        body = json.dumps({"model": MODEL, "messages": messages, "stream": False, "options": options, "format": "json"}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]["content"]


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def template_variables(template):
    # TODO 1: return the SET of {placeholder} names in a template string.
    #   "Hi {name}, you are {role}." -> {"name", "role"}
    # Hint: string.Formatter().parse(template) yields (literal, field_name, spec, conversion); field_name is None for plain text.
    placeholders = set()
    for part in string.Formatter().parse(template):
        field_name = part[1]
        if field_name is not None:
            placeholders.add(field_name)
    return placeholders


def build_messages(prompt, variables):
    # TODO 2: turn a prompt spec into a chat message list.
    #   prompt = {"system": <template or None>, "user": <template>}
    #   - If any placeholder in either template is missing from `variables`, raise KeyError
    #     whose message lists ALL missing names (sorted), e.g. KeyError("missing variables: ['a', 'b']").
    #     A silent "{filing}" left in a production prompt is a real, common bug.
    #   - Fill both templates with str.format(**variables).
    #   - Return [{"role": "system", ...}, {"role": "user", ...}], leaving out the system message when it's None.
    missing_var = set()
    if prompt.get("system") is not None:
        missing_var.update(template_variables(prompt["system"]) - variables.keys())
    missing_var.update(template_variables(prompt["user"]) - variables.keys())
    if missing_var:
        raise KeyError(f"missing variables: {sorted(missing_var)}")
    messages = []
    if prompt.get("system") is not None:
        messages.append({"role": "system", "content": prompt["system"].format(**variables)})
    messages.append({"role": "user", "content": prompt["user"].format(**variables)})
    return messages


def parse_json_strict(text):
    # Provided-ish helper you'll use in TODO 3: what a downstream parser sees. No cleanup, no fence stripping.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def score_response(text, expected):
    # TODO 3: score one reply for the extraction task.
    #   expected = {"company": ..., "form": ..., "fiscal_year": ...}   (fiscal_year is an int, or None when the text has none)
    # Return a dict:
    #   "valid_json":  the WHOLE reply parses as a JSON object (dict) with parse_json_strict
    #   "exact_keys":  valid_json and its keys are exactly the expected keys (no extras, none missing)
    #   "correct":     exact_keys and every value matches expected; compare strings case-insensitively
    #                  after .strip(); fiscal_year must equal the int (accept "2023" as 2023, but not "FY2023")
    # Each later check implies the earlier ones, so the three bools never go True after a False
    expected_keys = set(expected.keys())
    result = {"valid_json": False, "exact_keys": False, "correct": False}
    parsed = parse_json_strict(text)
    if isinstance(parsed, dict):
        result["valid_json"] = True
        if set(parsed.keys()) == expected_keys:
            result["exact_keys"] = True
            correct = True
            for key in expected_keys:
                expected_value = expected[key]
                parsed_value = parsed[key]
                if isinstance(expected_value, int):
                    try:
                        if isinstance(parsed_value, bool):
                            correct = False
                            break

                        if int(parsed_value) != expected_value:
                            correct = False
                            break

                    except (ValueError, TypeError, OverflowError):
                        correct = False
                        break
                elif isinstance(expected_value, str):
                    if not (isinstance(parsed_value, str) and parsed_value.strip().lower() == expected_value.strip().lower()):
                        correct = False
                        break
                elif expected_value is None:
                    if parsed_value is not None:
                        correct = False
                        break
                else:
                    correct = False
                    break
            result["correct"] = correct
    return result

def run_suite(prompt, cases, n_samples=3, temperature=0.7, llm=call_llm):
    # TODO 4: run one prompt over every case, n_samples times each, and aggregate.
    #   cases = [{"vars": {...}, "expected": {...}}, ...]
    #   For sample i use seed=i so runs are reproducible.
    # Return {"valid_json": rate, "exact_keys": rate, "correct": rate, "n": total_calls, "failures": [...]}
    #   where each rate is the fraction of all calls in [0, 1], and failures lists up to 5 raw
    #   replies that weren't "correct" (you'll read them to see HOW the prompt fails).
    total_calls = 0
    valid_json_count = 0
    exact_keys_count = 0
    correct_count = 0
    failures = []
    for case in cases:
        vars = case["vars"]
        expected = case["expected"]
        for i in range(n_samples):
            total_calls += 1
            messages = build_messages(prompt, vars)
            reply = llm(messages, temperature=temperature, seed=i)
            score = score_response(reply, expected)
            score = score_response(reply, expected)

            if score["valid_json"]:
                valid_json_count += 1

            if score["exact_keys"]:
                exact_keys_count += 1

            if score["correct"]:
                correct_count += 1

            if not score["correct"] and len(failures) < 5:
                failures.append(reply)
    return {
        "valid_json": valid_json_count / total_calls,
        "exact_keys": exact_keys_count / total_calls,
        "correct": correct_count / total_calls,
        "n": total_calls,
        "failures": failures
    }


# ---------------------------------------------------------------------------
# Self-checks (offline, with a fake LLM). Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


if __name__ == "__main__":
    ok = []
    ok.append(check("1 template_variables", template_variables("Hi {name}, you are {role}. {name}!") == {"name", "role"}))
    ok.append(check("1 no placeholders", template_variables("plain text") == set()))

    p = {"system": "You are {role}.", "user": "Read <doc>{doc}</doc>"}
    m = build_messages(p, {"role": "an analyst", "doc": "X"})
    ok.append(check("2 builds system + user", m == [{"role": "system", "content": "You are an analyst."},
                                                    {"role": "user", "content": "Read <doc>X</doc>"}], str(m)))
    ok.append(check("2 no system when None", build_messages({"system": None, "user": "{q}"}, {"q": "hi"}) ==
                    [{"role": "user", "content": "hi"}]))
    try:
        build_messages(p, {})
        ok.append(check("2 missing vars raise", False, "no error raised"))
    except KeyError as e:
        ok.append(check("2 missing vars listed", "doc" in str(e) and "role" in str(e), str(e)))

    exp = {"company": "Apple Inc.", "form": "10-K", "fiscal_year": 2023}
    good = '{"company": "apple inc. ", "form": "10-K", "fiscal_year": "2023"}'
    s = score_response(good, exp)
    ok.append(check("3 correct reply", s == {"valid_json": True, "exact_keys": True, "correct": True}, str(s)))
    s = score_response('Sure! Here is the JSON:\n{"company": "Apple Inc.", "form": "10-K", "fiscal_year": 2023}', exp)
    ok.append(check("3 preamble breaks the parser", s == {"valid_json": False, "exact_keys": False, "correct": False}, str(s)))
    s = score_response('{"company": "Apple Inc.", "form": "10-K", "fiscal_year": 2023, "note": "x"}', exp)
    ok.append(check("3 extra key", s == {"valid_json": True, "exact_keys": False, "correct": False}, str(s)))
    s = score_response('{"company": "Apple Inc.", "form": "10-K", "fiscal_year": "FY2023"}', exp)
    ok.append(check("3 FY2023 is wrong", s["exact_keys"] and not s["correct"], str(s)))
    s = score_response('{"company": "X", "form": "8-K", "fiscal_year": null}', {"company": "x", "form": "8-K", "fiscal_year": None})
    ok.append(check("3 null year matches None", s["correct"], str(s)))
    ok.append(check("3 JSON list is not an object", not score_response("[1, 2]", exp)["valid_json"]))

    replies = iter(['{"company": "A", "form": "10-K", "fiscal_year": 2023}', "A, 10-K, 2023"] * 2)
    seeds = []
    def fake_llm(messages, temperature=0.0, seed=None, max_tokens=200):
        seeds.append(seed)
        return next(replies)
    cases = [{"vars": {"role": "r", "doc": "d"}, "expected": {"company": "A", "form": "10-K", "fiscal_year": 2023}}] * 2
    r = run_suite(p, cases, n_samples=2, llm=fake_llm)
    ok.append(check("4 rates and n", r["n"] == 4 and r["valid_json"] == 0.5 and r["correct"] == 0.5, str(r)))
    ok.append(check("4 seeds 0..n-1 per case", seeds == [0, 1, 0, 1], str(seeds)))
    ok.append(check("4 failures keep raw replies", r["failures"] == ["A, 10-K, 2023", "A, 10-K, 2023"], str(r["failures"])))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
