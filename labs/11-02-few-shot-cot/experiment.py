# Lesson 11/02 "Use it": measure zero-shot vs CoT vs few-shot CoT vs self-consistency on real llama3.2.
# Run: python3 labs/11-02-few-shot-cot/experiment.py     (needs `ollama serve`; ~5-8 min)
# 12 original multi-step word problems, written for this lab. Accuracy, output tokens and wall time per technique.
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from my_reasoning import build_messages, call_llm, extract_answer, self_consistency  # your code

PROBLEMS = [
    ("A bakery makes 240 rolls. It sells 3/8 of them in the morning and 1/3 of the remaining rolls in the "
     "afternoon. How many rolls are left?", "100"),
    ("A phone plan costs $35 a month plus $0.10 for each text over 300 texts. Maya sent 420 texts this month. "
     "What is her bill in dollars?", "47"),
    ("A train leaves at 9:40 and arrives at 13:15 the same day. It stopped at stations for 25 minutes in total. "
     "For how many minutes was it moving?", "190"),
    ("Sam is 3 times as old as his daughter. In 12 years he will be twice as old as she will be. "
     "How old is Sam now?", "36"),
    ("A 1,200-liter tank is 35% full. A pump adds 45 liters per minute. How many minutes until the tank is 80% full?", "12"),
    ("A store marks up an $80 jacket by 25%, then puts it on sale at 20% off the new price. "
     "What is the sale price in dollars?", "80"),
    ("A rectangle's length is 4 cm more than twice its width. Its perimeter is 50 cm. What is its area in square cm?", "126"),
    ("Leo reads 18 pages on each weekday and 40 pages on each weekend day. How many pages does he read in 3 weeks?", "510"),
    ("A factory makes 450 widgets a day. 4% are defective and thrown away. The good ones are packed in boxes of 24. "
     "How many full boxes does it pack each day?", "18"),
    ("Ana has twice as many marbles as Ben. Ben has 7 more marbles than Cara. Together the three have 73 marbles. "
     "How many marbles does Ana have?", "40"),
    ("A car uses 6.5 liters of fuel per 100 km. Fuel costs $1.80 per liter. How much does fuel cost, in dollars, "
     "for a 340 km trip?", "39.78"),
    ("A company has 1,500 employees. It grows its headcount by 20%, and then 1/6 of the new total leave. "
     "How many employees does it have now?", "1500"),
]


def single(mode):
    # One call per problem at temperature 0 (greedy).
    correct, tokens, misses = 0, 0, []
    for q, gold in PROBLEMS:
        text, t = call_llm(build_messages(q, mode), temperature=0.0, seed=0)
        got = extract_answer(text)
        tokens += t
        if got == gold:
            correct += 1
        else:
            misses.append((gold, got, text))
    return correct, tokens, misses


def show(name, correct, tokens, secs, extra=""):
    n = len(PROBLEMS)
    print(f"{name:<22} {correct:>2}/{n} = {correct / n:4.0%}   tokens/question {tokens / n:6.0f}   {secs:5.0f}s  {extra}")


if __name__ == "__main__":
    print(f"{len(PROBLEMS)} problems, llama3.2 3B\n")
    all_misses = {}
    for mode in ("zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"):
        t0 = time.perf_counter()
        c, tok, misses = single(mode)
        show(mode, c, tok, time.perf_counter() - t0)
        all_misses[mode] = misses

    t0 = time.perf_counter()
    c, tok, low = 0, 0, []
    for q, gold in PROBLEMS:
        r = self_consistency(q, n=5, temperature=0.7)
        tok += r["tokens"]
        c += r["answer"] == gold
        low.append((r["confidence"], r["answer"] == gold, gold, r["votes"]))
    show("self_consistency n=5", c, tok, time.perf_counter() - t0, "(few_shot_cot, T=0.7)")

    print("\nfew_shot_cot misses (gold, extracted, start of reply):")
    for gold, got, text in all_misses["few_shot_cot"]:
        print(f"    {gold:>6}  {str(got):>8}   {text[:110]!r}")

    print("\nself-consistency votes, sorted by confidence (conf, right?, gold, votes):")
    for conf, right, gold, votes in sorted(low, key=lambda x: x[0]):
        print(f"    {conf:.1f}  {'ok ' if right else 'BAD'}  {gold:>6}  {votes}")
