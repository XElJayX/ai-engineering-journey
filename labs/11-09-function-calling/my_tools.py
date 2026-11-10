# Lesson 11/09 Function Calling & Tool Use: argument validation, a tool registry that never raises, and the tool loop.
# Run: python labs/11-09-function-calling/my_tools.py      (self-checks use a scripted fake model; no LLM needed)
# The "Use it" run (experiment.py) puts YOUR loop between llama3.2 (Ollama) and a fictional financial database,
# then tests closed-book vs tools, vague vs clear descriptions, error messages, and the iteration cap.
import json

# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def validate_args(parameters, args):
    # TODO 1: check `args` (a dict the MODEL wrote) against a JSON-Schema-style `parameters` object:
    #     {"type": "object", "properties": {name: {"type": ..., "enum": [...], "minimum": .., "maximum": ..}}, "required": [...]}
    #   Return a list of human-readable error strings ([] = valid). Check, in this order:
    #     a) args is a dict                          -> else ["arguments must be an object"] and stop
    #     b) every name in "required" is present     -> "missing required argument 'x'"
    #     c) no names outside "properties"           -> "unknown argument 'x'"
    #     d) each present value has the right type:  "string" -> str, "integer" -> int but NOT bool,
    #        "number" -> int or float but NOT bool, "boolean" -> bool
    #        -> "argument 'x' must be integer, got str"   (use type(value).__name__)
    #     e) only if the type is right: "enum" membership, "minimum" / "maximum" bounds
    #        -> "argument 'x' must be one of [...]" / "argument 'x' must be >= 2015" / "... <= 2025"
    #   Gotcha: in Python, isinstance(True, int) is True. The model writing true for a year must be rejected.
    errors = []

    if not isinstance(args, dict):
        return ["arguments must be an object"]

    required = parameters.get("required",[])

    for require in required:
        if require not in args:
            errors.append(f"missing required argument '{require}'")
    
    properties = parameters.get("properties", {})
    for key in args.keys():
        if key not in properties:
            errors.append(f"unknown argument '{key}'")
    
    for key, value in args.items():
        if key not in properties:
            continue
        type_valid = True
        schema = properties[key]
        expected_type = schema.get("type")

        match expected_type:
            case "string":
                if not isinstance(value, str):
                    errors.append(f"argument '{key}' must be {expected_type}, got {type(value).__name__}")
                    type_valid = False
            case "integer":
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"argument '{key}' must be {expected_type}, got {type(value).__name__}")
                    type_valid = False
            case "number":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"argument '{key}' must be {expected_type}, got {type(value).__name__}")
                    type_valid = False
            case "boolean":
                if not isinstance(value, bool):
                    errors.append(f"argument '{key}' must be {expected_type}, got {type(value).__name__}")
                    type_valid = False
            case _:
                errors.append(f"argument '{key}' has unknown type '{expected_type}' in schema")
                type_valid = False
        if not type_valid:
            continue  # skip enum/min/max checks if type is wrong
        if "enum" in schema:
            expected_enums = schema.get("enum")
            if value not in expected_enums:
                errors.append(f"argument '{key}' must be one of {expected_enums}")
        if expected_type in ["integer", "number"]:
            if "minimum" in schema:
                minimum = schema.get("minimum")
                if value < minimum:
                    errors.append(f"argument '{key}' must be >= {minimum}")
            if "maximum" in schema:
                maximum = schema.get("maximum")
                if value > maximum:
                    errors.append(f"argument '{key}' must be <= {maximum}")

            
            

    return errors


class ToolRegistry:
    def __init__(self):
        self.tools = {}   # name -> {"fn": callable, "description": str, "parameters": dict}

    def register(self, name, fn, description, parameters):
        self.tools[name] = {"fn": fn, "description": description, "parameters": parameters}

    def schemas(self):
        # TODO 2a: return the list the API wants, one entry per tool, in registration order:
        #     {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        results = []
        for name, tool in self.tools.items():
            results.append(
                {
                    "type": "function",
                    "function":{
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
            )
        return results



    def call(self, name, args):
        # TODO 2b: run a tool the model asked for and ALWAYS return a dict (never raise: the loop must survive).
        #     unknown name           -> {"error": "UNKNOWN_TOOL", "message": "no tool named 'x'; available: a, b"}
        #     validate_args fails    -> {"error": "INVALID_ARGS", "message": "; ".join(errors)}   (do NOT run fn)
        #     fn raises exception e  -> {"error": "TOOL_FAILED", "message": f"{type(e).__name__}: {e}"}
        #     otherwise              -> fn(**args)  (tools return dicts)
        #   Allowlist: only registered names can ever run. Nothing like getattr / eval on the name.
        available_fn = ", ".join(self.tools.keys())
        if name not in self.tools:
            return {"error": "UNKNOWN_TOOL","message": f"no tool named '{name}'; available: {available_fn}"}
        parameters = self.tools[name]["parameters"]
        errors = validate_args(parameters= parameters, args = args)
        if errors:
            return {"error": "INVALID_ARGS", "message": "; ".join(errors)}
        try:
            function = self.tools[name]["fn"]
            result = function(**args)
        except Exception as e:
            return {"error": "TOOL_FAILED", "message": f"{type(e).__name__}: {e}"}
        return result




        

def run_loop(chat, messages, registry, max_iters=5):
    # TODO 3: the function-calling loop.
    #   chat(messages, tools) -> an assistant message dict, Ollama/OpenAI style:
    #       {"role": "assistant", "content": "...", "tool_calls": [{"function": {"name": "...", "arguments": {...}}}, ...]}
    #   ("tool_calls" may be missing or empty; "arguments" may be a dict OR a JSON string; handle both).
    #   Repeat up to max_iters times:
    #       msg = chat(messages, registry.schemas());  append msg to messages
    #       no tool calls -> return (msg["content"], trace)
    #       else for EACH call (parallel calls come as several entries):
    #           args = the arguments (json.loads if str; if that fails -> result {"error": "BAD_JSON", "message": str(e)})
    #           result = registry.call(name, args)  (unless BAD_JSON)
    #           trace.append({"name": name, "args": args, "result": result})
    #           messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    #   If the loop runs out: return ("STOPPED: hit max_iters", trace). Never loop forever.
    #   Don't copy `messages`: append to the list you were given (the caller wants the full transcript).
    trace =[]

    for _ in range(max_iters):
        output = chat(messages, registry.schemas())
        messages.append(output)

        tool_calls = output.get("tool_calls",[])
        if not tool_calls:
            return (output["content"],trace)

        for tool_call in tool_calls:
            func_dict = tool_call["function"]
            tool_name = func_dict["name"]
            args = func_dict["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as e:
                    result = {"error": "BAD_JSON", "message": str(e)}
                else:
                    result = registry.call(tool_name, args)
            else:
                result = registry.call(tool_name, args)
            trace.append({"name": tool_name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_name": tool_name, "content": json.dumps(result)})
    return ("STOPPED: hit max_iters", trace)



# ---------------------------------------------------------------------------
# Self-checks. Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


REV_PARAMS = {"type": "object",
              "properties": {"company": {"type": "string"},
                             "fiscal_year": {"type": "integer", "minimum": 2015, "maximum": 2025},
                             "units": {"type": "string", "enum": ["usd", "usd_millions"]}},
              "required": ["company", "fiscal_year"]}


def scripted(*replies):
    # a fake model that returns canned assistant messages and records what it was shown
    seen = []

    def chat(messages, tools):
        seen.append((len(messages), [t["function"]["name"] for t in tools]))
        return replies[min(len(seen), len(replies)) - 1]
    return chat, seen


def tc(name, args):
    return {"function": {"name": name, "arguments": args}}


if __name__ == "__main__":
    ok = []
    v = validate_args
    ok.append(check("1 valid args -> []", v(REV_PARAMS, {"company": "Halcyon", "fiscal_year": 2024}) == []))
    ok.append(check("1 not a dict", v(REV_PARAMS, ["Halcyon", 2024]) == ["arguments must be an object"]))
    ok.append(check("1 missing required", v(REV_PARAMS, {"company": "Halcyon"}) == ["missing required argument 'fiscal_year'"],
                    str(v(REV_PARAMS, {"company": "Halcyon"}))))
    ok.append(check("1 unknown argument", v(REV_PARAMS, {"company": "H", "fiscal_year": 2024, "sql": "DROP"}) == ["unknown argument 'sql'"]))
    ok.append(check("1 wrong type ('FY2024' as a string)", v(REV_PARAMS, {"company": "H", "fiscal_year": "FY2024"}) == ["argument 'fiscal_year' must be integer, got str"],
                    str(v(REV_PARAMS, {"company": "H", "fiscal_year": "FY2024"}))))
    ok.append(check("1 bool is not an integer", v(REV_PARAMS, {"company": "H", "fiscal_year": True}) == ["argument 'fiscal_year' must be integer, got bool"]))
    ok.append(check("1 bounds", v(REV_PARAMS, {"company": "H", "fiscal_year": 1999}) == ["argument 'fiscal_year' must be >= 2015"]
                    and v(REV_PARAMS, {"company": "H", "fiscal_year": 2031}) == ["argument 'fiscal_year' must be <= 2025"]))
    ok.append(check("1 enum", v(REV_PARAMS, {"company": "H", "fiscal_year": 2024, "units": "eur"}) == ["argument 'units' must be one of ['usd', 'usd_millions']"],
                    str(v(REV_PARAMS, {"company": "H", "fiscal_year": 2024, "units": "eur"}))))
    ok.append(check("1 number accepts int and float, not bool", v({"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}, {"x": 2}) == []
                    and v({"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}, {"x": 2.5}) == []
                    and len(v({"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}, {"x": False})) == 1))

    reg = ToolRegistry()
    calls = []

    def get_revenue(company, fiscal_year, units="usd_millions"):
        calls.append(company)
        if company == "boom":
            raise ConnectionError("db timeout")
        return {"company": company, "fiscal_year": fiscal_year, "revenue_musd": 1284}
    reg.register("get_revenue", get_revenue, "Revenue for a company and fiscal year.", REV_PARAMS)
    reg.register("ping", lambda: {"ok": True}, "Health check.", {"type": "object", "properties": {}, "required": []})
    sch = reg.schemas()
    ok.append(check("2a schemas: API shape, registration order", [s["type"] for s in sch] == ["function", "function"]
                    and [s["function"]["name"] for s in sch] == ["get_revenue", "ping"] and sch[0]["function"]["parameters"] is REV_PARAMS))
    ok.append(check("2b good call runs the tool", reg.call("get_revenue", {"company": "H", "fiscal_year": 2024})["revenue_musd"] == 1284))
    r = reg.call("delete_all", {})
    ok.append(check("2b unknown tool -> UNKNOWN_TOOL listing what exists", r.get("error") == "UNKNOWN_TOOL" and "get_revenue" in r.get("message", ""), str(r)))
    n_before = len(calls)
    r = reg.call("get_revenue", {"company": "H", "fiscal_year": "2024"})
    ok.append(check("2b invalid args -> INVALID_ARGS and the tool is NOT run", r.get("error") == "INVALID_ARGS" and len(calls) == n_before
                    and "must be integer" in r.get("message", ""), str(r)))
    r = reg.call("get_revenue", {"company": "boom", "fiscal_year": 2024})
    ok.append(check("2b tool exception -> TOOL_FAILED, no raise", r == {"error": "TOOL_FAILED", "message": "ConnectionError: db timeout"}, str(r)))

    chat, seen = scripted({"role": "assistant", "content": "Revenue was $1,284M."})
    msgs = [{"role": "user", "content": "hi"}]
    ans, trace = run_loop(chat, msgs, reg)
    ok.append(check("3 no tool call -> answer directly, 1 model call", ans == "Revenue was $1,284M." and trace == [] and len(seen) == 1 and len(msgs) == 2))
    chat, seen = scripted({"role": "assistant", "content": "", "tool_calls": [tc("get_revenue", {"company": "A", "fiscal_year": 2024}),
                                                                             tc("get_revenue", '{"company": "B", "fiscal_year": 2023}')]},
                          {"role": "assistant", "content": "A: 1284, B: 1284"})
    msgs = [{"role": "user", "content": "A and B?"}]
    ans, trace = run_loop(chat, msgs, reg)
    ok.append(check("3 parallel calls: both run, JSON-string args parsed, 2 model calls", ans == "A: 1284, B: 1284" and [t["args"]["company"] for t in trace] == ["A", "B"]
                    and len(seen) == 2, f"trace {trace}"))
    ok.append(check("3 transcript: user, assistant, tool, tool, assistant", [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool", "assistant"],
                    str([m["role"] for m in msgs])))
    ok.append(check("3 tool messages carry JSON results the model can read", json.loads(msgs[2]["content"])["revenue_musd"] == 1284 and msgs[2]["tool_name"] == "get_revenue"))
    ok.append(check("3 the model sees tool schemas every call", seen[0][1] == ["get_revenue", "ping"]))
    chat, _ = scripted({"role": "assistant", "content": "", "tool_calls": [tc("get_revenue", "{company: A}")]}, {"role": "assistant", "content": "sorry"})
    msgs = [{"role": "user", "content": "x"}]
    ans, trace = run_loop(chat, msgs, reg)
    ok.append(check("3 malformed JSON args -> BAD_JSON result, loop continues", trace and trace[0]["result"].get("error") == "BAD_JSON" and ans == "sorry", str(trace)))
    chat, seen = scripted({"role": "assistant", "content": "", "tool_calls": [tc("ping", {})]})   # model that never stops calling
    ans, trace = run_loop(chat, [{"role": "user", "content": "x"}], reg, max_iters=4)
    ok.append(check("3 never-ending tool calls stop at max_iters", ans == "STOPPED: hit max_iters" and len(seen) == 4 and len(trace) == 4, f"{ans!r}, {len(seen)} calls"))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
