# Lesson 11/09 "Use it": YOUR validator, registry and loop between llama3.2 (Ollama, native tool calls) and a fictional
# financial database. The companies are invented, so the model cannot know the numbers from training.
# Run: python labs/11-09-function-calling/experiment.py      (needs Ollama running with llama3.2; ~2 min)
#   A. closed-book vs tools on 10 questions (5 lookups + 5 growth %), with YOUR strict validator vs a lenient
#      one that first coerces "2024" -> 2024 (what Pydantic-style "lax" validation does)
#   B. the SAME tools with vague names/descriptions ("tool_a", "Data.", params x/y/z) vs clear ones
#   C. strict validator, how the error is reported: bare {"error": "INVALID_ARGS"} vs YOUR message + "call it again"
#   D. a tool that always says "not ready, call again": does YOUR max_iters stop it?
#   + every INVALID_ARGS your validator caught (wrong types the model actually wrote)
import difflib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_tools import ToolRegistry, run_loop  # noqa: E402  (your code)

MODEL = "llama3.2"
URL = "http://localhost:11434/api/chat"
SYSTEM = ("You are a financial assistant. Use the tools to look up numbers; never guess a figure. "
          "Answer in one sentence with the number.")

DB = {  # fiscal year -> (revenue, net income), USD millions. All fictional.
    "Halcyon Robotics": {2022: (1012, 88), 2023: (1150, 97), 2024: (1284, 131)},
    "Vantor Biotech": {2022: (402, -35), 2023: (455, -12), 2024: (533, 21)},
    "Kestrel Freight": {2022: (2210, 140), 2023: (2087, 96), 2024: (2342, 158)},
    "Orlis Foods": {2022: (876, 54), 2023: (921, 60), 2024: (948, 49)},
    "Brightmere Energy": {2022: (3105, 410), 2023: (2890, 302), 2024: (3377, 455)},
}

QUESTIONS = [  # (question, expected number, kind)
    ("What was Halcyon Robotics' revenue in fiscal 2024?", 1284, "value"),
    ("What was Vantor Biotech's net income in fiscal 2023?", -12, "value"),
    ("What was Kestrel Freight's revenue in fiscal 2022?", 2210, "value"),
    ("What was Orlis Foods' net income in fiscal 2024?", 49, "value"),
    ("What was Brightmere Energy's revenue in fiscal 2023?", 2890, "value"),
    ("By what percent did Halcyon Robotics' revenue grow from fiscal 2023 to fiscal 2024?", 11.65, "pct"),
    ("By what percent did Kestrel Freight's revenue change from fiscal 2022 to fiscal 2023?", -5.57, "pct"),
    ("By what percent did Brightmere Energy's net income change from fiscal 2023 to fiscal 2024?", 50.66, "pct"),
    ("By what percent did Vantor Biotech's revenue grow from fiscal 2022 to fiscal 2024?", 32.59, "pct"),
    ("By what percent did Orlis Foods' revenue grow from fiscal 2023 to fiscal 2024?", 2.93, "pct"),
]


N_CALLS = {"n": 0}


def chat(messages, tools):
    N_CALLS["n"] += 1
    body = {"model": MODEL, "messages": messages, "stream": False, "options": {"temperature": 0, "num_ctx": 4096}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]


# ---- the tools (the code the model can never see, only call) ----

class LenientRegistry(ToolRegistry):
    """Coerce obvious type slips ("2024" -> 2024, "1150.0" -> 1150.0) BEFORE your validator runs."""
    coerced = 0

    def call(self, name, args):
        if name in self.tools and isinstance(args, dict):
            props = self.tools[name]["parameters"].get("properties", {})
            args = dict(args)
            for k, v in args.items():
                t = props.get(k, {}).get("type")
                if isinstance(v, str) and t in ("integer", "number") and re.fullmatch(r"-?\d+(\.\d+)?", v.strip()):
                    args[k] = int(float(v)) if t == "integer" else float(v)
                    LenientRegistry.coerced += 1
        return super().call(name, args)


class ErrorStyleRegistry(ToolRegistry):
    """Same validator; only changes how INVALID_ARGS is reported back to the model."""
    def __init__(self, style):
        super().__init__()
        self.style = style

    def call(self, name, args):
        r = super().call(name, args)
        if isinstance(r, dict) and r.get("error") == "INVALID_ARGS":
            if self.style == "bare":
                return {"error": "INVALID_ARGS"}
            return dict(r, next_step=f"Fix exactly these arguments and call {name} again.")
        return r


def make_registry(clear=True, helpful_errors=True, cls=ToolRegistry, **kw):
    def find(company):
        if company in DB:
            return company, None
        if helpful_errors:
            close = difflib.get_close_matches(company, list(DB), n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            return None, {"error": "COMPANY_NOT_FOUND", "message": f"No company named '{company}'.{hint} Known: {', '.join(DB)}."}
        return None, {"error": "NOT_FOUND"}

    def get_financial(company, fiscal_year, metric):
        name, err = find(company)
        if err:
            return err
        if fiscal_year not in DB[name]:
            return {"error": "YEAR_NOT_FOUND", "message": f"{name} has fiscal years {sorted(DB[name])}."}
        rev, ni = DB[name][fiscal_year]
        return {"company": name, "fiscal_year": fiscal_year, "metric": metric,
                "value_usd_millions": rev if metric == "revenue" else ni}

    def pct_change(old, new):
        if old == 0:
            return {"error": "DIVIDE_BY_ZERO", "message": "old value is 0"}
        return {"pct_change": round((new - old) / abs(old) * 100, 2)}

    reg = cls(**kw)
    if clear:
        reg.register("get_financial", get_financial,
                     "Look up one reported figure for a company and fiscal year from the 10-K database. "
                     "Returns the value in USD millions.",
                     {"type": "object",
                      "properties": {"company": {"type": "string", "description": "Exact company name, e.g. 'Halcyon Robotics'"},
                                     "fiscal_year": {"type": "integer", "description": "Fiscal year as a number, e.g. 2024"},
                                     "metric": {"type": "string", "enum": ["revenue", "net_income"]}},
                      "required": ["company", "fiscal_year", "metric"]})
        reg.register("pct_change", pct_change,
                     "Percent change from an old value to a new value: (new - old) / |old| * 100. Use it for growth questions.",
                     {"type": "object",
                      "properties": {"old": {"type": "number", "description": "Earlier value"},
                                     "new": {"type": "number", "description": "Later value"}},
                      "required": ["old", "new"]})
    else:   # same functions, same behaviour; only the names and descriptions are vague
        reg.register("tool_a", lambda x, y, z: get_financial(x, y, z), "Data.",
                     {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "integer"},
                                                       "z": {"type": "string", "enum": ["revenue", "net_income"]}},
                      "required": ["x", "y", "z"]})
        reg.register("tool_b", lambda x, y: pct_change(x, y), "Math.",
                     {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]})
    return reg


def correct(answer, want, kind):
    a = answer.replace(",", "").replace("−", "-")
    if kind == "pct":
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?=\s*%)", a)]
        return any(abs(abs(n) - abs(want)) <= 0.15 for n in nums)
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    return any(abs(abs(n) - abs(want)) < 0.5 or abs(abs(n) * 1000 - abs(want)) < 6 for n in nums)


def states_number(answer):
    return bool(re.search(r"\$\s?\d|\d+(?:\.\d+)?\s*(?:%|million|billion)", answer))


def run(questions, reg, use_tools=True, max_iters=6):
    out = []
    for q, want, kind in questions:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        before = N_CALLS["n"]
        if use_tools:
            ans, trace = run_loop(chat, msgs, reg, max_iters=max_iters)
        else:
            ans, trace = chat(msgs, None).get("content", ""), []
        out.append({"q": q, "ans": ans, "ok": correct(ans, want, kind), "trace": trace, "calls": N_CALLS["n"] - before})
    return out


def summary(label, rows):
    ok = sum(r["ok"] for r in rows)
    look = sum(r["ok"] for r, (_, _, k) in zip(rows, QUESTIONS) if k == "value") if len(rows) == len(QUESTIONS) else None
    wrong_num = sum((not r["ok"]) and states_number(r["ans"]) for r in rows)
    tools = sum(len(r["trace"]) for r in rows)
    errs = sum(1 for r in rows for t in r["trace"] if isinstance(t["result"], dict) and "error" in t["result"])
    calls = sum(r["calls"] for r in rows)
    split = f" (lookups {look}/5, growth {ok - look}/5)" if look is not None else ""
    print(f"   {label:<28} correct {ok:>2}/{len(rows)}{split}   wrong number stated {wrong_num:>2}   "
          f"tool calls {tools:>3}   tool errors {errs:>2}   model calls {calls:>3}")


def invalid_args(rows):
    return [(t["name"], t["args"], t["result"].get("message", "(bare: no message)")) for r in rows for t in r["trace"]
            if isinstance(t["result"], dict) and t["result"].get("error") == "INVALID_ARGS"]


if __name__ == "__main__":
    t0 = time.time()
    try:
        chat([{"role": "user", "content": "hi"}], None)
    except Exception as e:
        sys.exit(f"Ollama not reachable at {URL}: {e}. Start it with `ollama serve`.")
    clear = make_registry()
    print(f"model {MODEL}   {len(DB)} fictional companies   tools: {[s['function']['name'] for s in clear.schemas()]}")

    print("\n== A. Closed-book vs tools (10 questions: 5 lookups, 5 growth %) ==")
    closed = run(QUESTIONS, None, use_tools=False)
    strict = run(QUESTIONS, clear)
    lenient = run(QUESTIONS, make_registry(cls=LenientRegistry))
    summary("closed-book (no tools)", closed)
    summary("tools, strict validator", strict)
    summary("tools, lenient (coerce)", lenient)
    print(f"   closed-book example: {closed[0]['ans'][:140]!r}")
    print(f"   lenient registry coerced {LenientRegistry.coerced} string numbers before validating")
    for label, r in (("lookup", lenient[0]), ("growth", lenient[5])):
        print(f"   {label} trace (lenient): {r['q']}")
        for t in r["trace"][:4]:
            print(f"      -> {t['name']}({json.dumps(t['args'])})  =>  {json.dumps(t['result'])[:120]}")
        print(f"      answer: {r['ans'][:140]!r}")

    print("\n== B. Same tools (lenient), vague schemas ('tool_a': 'Data.', params x/y/z) ==")
    vague = run(QUESTIONS, make_registry(clear=False, cls=LenientRegistry))
    summary("clear schemas", lenient)
    summary("vague schemas", vague)
    bad = next((r for r in vague if not r["ok"]), None)
    if bad:
        print(f"   a vague-schema miss: {bad['q']}")
        for t in bad["trace"][:3]:
            print(f"      -> {t['name']}({json.dumps(t['args'])})  =>  {json.dumps(t['result'])[:110]}")
        print(f"      answer: {bad['ans'][:140]!r}")

    print("\n== C. Strict validator: how should INVALID_ARGS be reported? (5 lookup questions) ==")
    lookups = QUESTIONS[:5]
    bare = run(lookups, make_registry(cls=ErrorStyleRegistry, style="bare"))
    action = run(lookups, make_registry(cls=ErrorStyleRegistry, style="actionable"))
    summary('bare {"error": "INVALID_ARGS"}', bare)
    summary("message + 'call it again'", action)
    retried = lambda rows: sum(len(r["trace"]) > 1 for r in rows)  # noqa: E731
    print(f"   questions where the model called the tool again after an error: bare {retried(bare)}/5, actionable {retried(action)}/5")
    r = action[0]
    for t in r["trace"][:3]:
        print(f"      -> {t['name']}({json.dumps(t['args'])})  =>  {json.dumps(t['result'])[:140]}")
    print(f"      answer: {r['ans'][:140]!r}")

    print("\n== D. A tool that never finishes: 'report not ready, call get_report again' ==")
    stuck = ToolRegistry()
    stuck.register("get_report", lambda company: {"status": "pending", "message": "Report is not ready yet. Call get_report again."},
                   "Fetch the full annual report for a company. May need to be called again if it is not ready.",
                   {"type": "object", "properties": {"company": {"type": "string"}}, "required": ["company"]})
    for sys_prompt in ("Use tools to get the data the user asks for.",
                       "Use tools to get the data the user asks for. If a tool says it is not ready, call it again until it is ready. Never make up data."):
        before = N_CALLS["n"]
        ans, trace = run_loop(chat, [{"role": "system", "content": sys_prompt},
                                     {"role": "user", "content": "Get me the full annual report for Halcyon Robotics."}], stuck, max_iters=8)
        print(f"   prompt {'plain ' if 'until' not in sys_prompt else '+retry'}: model calls {N_CALLS['n'] - before}, tool calls {len(trace)}, "
              f"result: {ans[:100]!r}")

    caught = invalid_args(strict + bare + action)
    print(f"\n== Your validator caught {len(caught)} invalid tool calls (the tool never ran) ==")
    for name, args, msg in caught[:6]:
        print(f"   {name}({json.dumps(args)})  ->  {msg}")
    print(f"\ntotal model calls {N_CALLS['n']}, {time.time() - t0:.0f}s")
