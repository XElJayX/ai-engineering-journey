# Lesson 11/02 Few-Shot, Chain-of-Thought, Self-Consistency: prompt modes, answer extraction, majority vote.
# Run: python3 labs/11-02-few-shot-cot/my_reasoning.py     (stdlib only; self-checks need no model)
# The live experiment (experiment.py) calls llama3.2 through Ollama at http://localhost:11434.
import json
import re
import urllib.request
from collections import Counter

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

# Three worked examples for the few-shot modes. Each has the question, a reasoning chain and the final answer.
EXAMPLES = [
    {"question": "A school orders 14 boxes of pencils with 36 pencils in each box. It gives 25 pencils to each "
                 "of 18 classrooms. How many pencils are left over?",
     "reasoning": "The school has 14 * 36 = 504 pencils. It gives away 18 * 25 = 450 pencils. "
                  "504 - 450 = 54 pencils are left.",
     "answer": "54"},
    {"question": "A shirt costs $40. It is discounted by 15%, and then a $5 coupon is applied. What is the final price in dollars?",
     "reasoning": "15% of $40 is 0.15 * 40 = $6, so the discounted price is 40 - 6 = $34. "
                  "The coupon takes off $5 more: 34 - 5 = $29.",
     "answer": "29"},
    {"question": "Priya is 4 years older than Omar. In 6 years, the sum of their ages will be 50. How old is Omar now?",
     "reasoning": "Let Omar be x now, so Priya is x + 4. In 6 years they are x + 6 and x + 10, "
                  "which sum to 2x + 16 = 50. So 2x = 34 and x = 17.",
     "answer": "17"},
]


# ---------------------------------------------------------------------------
# Provided: one LLM call through Ollama. Returns (reply_text, output_tokens) so you can measure cost.
# ---------------------------------------------------------------------------

def call_llm(messages, temperature=0.0, seed=None, max_tokens=400):
    options = {"temperature": temperature, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = seed
    body = json.dumps({"model": MODEL, "messages": messages, "stream": False, "options": options}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data["message"]["content"], data.get("eval_count", 0)


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def extract_answer(text):
    # TODO 1: pull the final numeric answer out of a reply, as a normalised string. Return None if there isn't one.
    #   - Use the LAST "the answer is <number>" in the text (case-insensitive). Models often say
    #     "the answer is" mid-reasoning and then correct themselves.
    #   - Allow an optional "$" before the number, and commas inside it: "The answer is $1,327.50." -> "1327.5"
    #   - Normalise: drop commas and a trailing ".0"/".50"-style zero tail, so "18.00" -> "18" and "39.780" -> "39.78".
    #     (Hint: float() then format; or strip zeros after a "." and then a dangling ".")
    #   - If there's no "answer is", accept the reply ONLY if the whole stripped reply is a single number
    #     (that's what an answer-only prompt returns). Do NOT fall back to "the last number in the text":
    #     on a reasoning reply that silently grades a random intermediate value.
    text = text.strip()
    # Case-insensitive search for "the answer is <number>"
    match = re.findall(r'answer is\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match: 
        answer = match[-1]
    else:
        answer = text.strip()
        if not re.fullmatch(r'-?\$?[\d,]+(?:\.\d+)?', answer):
            return None
    answer = answer.replace(",","")
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".") 
    return answer



def build_messages(question, mode, examples=EXAMPLES):
    # TODO 2: build the chat messages for one of four modes. Return [system, user] dicts.
    #   Every mode's system message must tell the model the output format, because extract_answer depends on it:
    #     "zero_shot":      answer with ONLY the final number, nothing else.
    #     "zero_shot_cot":  think step by step, then end with the line "The answer is <number>."
    #     "few_shot":       like zero_shot, but the user message first shows every example as
    #                       "Q: <question>\nA: The answer is <answer>.\n\n" (no reasoning), then "Q: <question>\nA:"
    #     "few_shot_cot":   like zero_shot_cot, and the examples include reasoning:
    #                       "Q: <question>\nA: <reasoning> The answer is <answer>.\n\n", then "Q: <question>\nA:"
    #   For the two zero-shot modes the user message is just the question.
    #   Raise ValueError for an unknown mode.
    match mode:
        case "zero_shot":
            system_msg = "Answer with ONLY the final number, nothing else."
            user_msg = question
        case "zero_shot_cot":
            system_msg = "Think step by step, then end with the line 'The answer is <number>.'"
            user_msg = question
        case "few_shot":
            system_msg = "Answer with ONLY the final number, nothing else."
            user_msg = "\n\n".join([f"Q: {ex['question']}\nA: The answer is {ex['answer']}." for ex in examples]) + f"\n\nQ: {question}\nA:"
        case "few_shot_cot":
            system_msg = "Think step by step, then end with the line 'The answer is <number>.'"
            user_msg = "\n\n".join([f"Q: {ex['question']}\nA: {ex['reasoning']} The answer is {ex['answer']}." for ex in examples]) + f"\n\nQ: {question}\nA:"
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

def majority_vote(answers):
    # TODO 3: answers is a list of extracted answers, which may contain None (unparseable replies).
    #   Return (winner, confidence):
    #     winner     = the most common non-None answer; on a tie, the one that appeared FIRST in the list
    #     confidence = winner's count / len(answers)   (Nones count in the denominator: a reply you
    #                  couldn't parse is a vote you didn't get)
    #   All None (or empty list) -> (None, 0.0)
    valid = [a for a in answers if a is not None]
    if not valid:
        return (None,0.0)
    counts = Counter(valid)
    return (counts.most_common(1)[0][0], counts.most_common(1)[0][1] / len(answers))


def self_consistency(question, n=5, temperature=0.7, llm=call_llm, mode="few_shot_cot"):
    # TODO 4: sample n replies (seed=i for sample i), extract each answer, and vote.
    #   Return {"answer": winner, "confidence": conf, "votes": the list of extracted answers, "tokens": total output tokens}
    #   llm(messages, temperature=..., seed=...) returns (text, n_output_tokens).
    answers=list()
    total_output_tokens = 0
    messages = build_messages(question,mode)
    for i in range(n):
        output, tokens = llm(messages,temperature,i)
        extracted = extract_answer(output)
        answers.append(extracted)
        total_output_tokens+= tokens
    winner, confidence = majority_vote(answers)
    return {"answer": winner, "confidence": confidence, "votes": answers, "tokens": total_output_tokens }
        


# ---------------------------------------------------------------------------
# Self-checks (offline, with a fake LLM). Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


if __name__ == "__main__":
    ok = []
    ok.append(check("1 basic", extract_answer("3 + 4 = 7. The answer is 7.") == "7"))
    ok.append(check("1 last one wins", extract_answer("So the answer is 12. Wait, I missed a step: the answer is 15.") == "15"))
    ok.append(check("1 dollars, commas, zeros", extract_answer("The answer is $1,327.50") == "1327.5",
                    repr(extract_answer("The answer is $1,327.50"))))
    ok.append(check("1 18.00 -> 18", extract_answer("the answer is 18.00.") == "18", repr(extract_answer("the answer is 18.00."))))
    ok.append(check("1 bare number reply", extract_answer("  42 \n") == "42"))
    ok.append(check("1 no last-number fallback", extract_answer("First 5 * 3 = 15, then 15 + 2 = 17") is None,
                    repr(extract_answer("First 5 * 3 = 15, then 15 + 2 = 17"))))
    ok.append(check("1 nothing", extract_answer("I am not sure.") is None))

    q = "How many legs do 3 spiders have?"
    m = build_messages(q, "zero_shot")
    ok.append(check("2 zero_shot shape", len(m) == 2 and m[0]["role"] == "system" and m[1] == {"role": "user", "content": q}, str(m)))
    fs = build_messages(q, "few_shot")[1]["content"]
    ok.append(check("2 few_shot: answers, no reasoning", fs.endswith(f"Q: {q}\nA:") and "The answer is 54." in fs
                    and EXAMPLES[0]["reasoning"] not in fs, repr(fs[-120:])))
    fsc = build_messages(q, "few_shot_cot")[1]["content"]
    ok.append(check("2 few_shot_cot: reasoning before answer",
                    f"A: {EXAMPLES[1]['reasoning']} The answer is 29.\n\n" in fsc and fsc.endswith(f"Q: {q}\nA:"), repr(fsc[:200])))
    sysmsgs = {md: build_messages(q, md)[0]["content"].lower() for md in ("zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot")}
    ok.append(check("2 CoT system prompts ask for 'the answer is'", "the answer is" in sysmsgs["zero_shot_cot"] and "the answer is" in sysmsgs["few_shot_cot"]))
    ok.append(check("2 modes differ", len(set(sysmsgs.values())) >= 2))
    try:
        build_messages(q, "tree_of_thought")
        ok.append(check("2 unknown mode raises", False, "no error"))
    except ValueError:
        ok.append(check("2 unknown mode raises", True))

    ok.append(check("3 plain majority", majority_vote(["24", "24", "27", "24", "18"]) == ("24", 0.6), str(majority_vote(["24", "24", "27", "24", "18"]))))
    ok.append(check("3 None counts against", majority_vote(["5", None, None, "5", "7"]) == ("5", 0.4), str(majority_vote(["5", None, None, "5", "7"]))))
    ok.append(check("3 tie -> first seen", majority_vote(["9", "3", "3", "9"])[0] == "9", str(majority_vote(["9", "3", "3", "9"]))))
    ok.append(check("3 all None", majority_vote([None, None]) == (None, 0.0) and majority_vote([]) == (None, 0.0)))

    replies = {0: ("... The answer is 10.", 30), 1: ("... The answer is 12.", 40), 2: ("... The answer is 10.", 50), 3: ("hmm", 5)}
    seen = []
    def fake_llm(messages, temperature=0.0, seed=None, max_tokens=400):
        seen.append((seed, temperature))
        return replies[seed]
    r = self_consistency("q?", n=4, temperature=0.9, llm=fake_llm)
    ok.append(check("4 vote", r["answer"] == "10" and r["confidence"] == 0.5 and r["votes"] == ["10", "12", "10", None], str(r)))
    ok.append(check("4 seeds and temperature", seen == [(0, 0.9), (1, 0.9), (2, 0.9), (3, 0.9)], str(seen)))
    ok.append(check("4 tokens summed", r["tokens"] == 125, str(r.get("tokens"))))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
