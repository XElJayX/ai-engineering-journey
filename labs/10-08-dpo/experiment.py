# Lesson 10/08 "Use it": preference-tune GPT-2 small (124M) with YOUR DPO, on the Apple GPU.
# Run: python labs/10-08-dpo/experiment.py      (needs transformers; GPT-2 is cached; ~15 min)
#   A. SFT (your 10/06 code) on answers written in TWO styles, concise and chatty -> the model does both. This is pi_ref.
#   B. Preference: humans prefer concise. Teach it 7 ways from the same SFT model:
#      SFT on the winners only / DPO beta 0.01, 0.1, 1.0 at lr 5e-7 / DPO at 10x the lr (5e-6) /
#      DPO + an SFT term on the winners (lr 5e-6) / SimPO (no reference model)
#      measured on UNSEEN prompts: % concise (sampled), % correct, English perplexity, held-out reward accuracy
#   C. the DPO lr 5e-6 run step by step: loss, held-out accuracy, and what happens to log pi(WINNER) itself
import copy
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "10-06-instruction-tuning-sft"))
from my_dpo import dpo_loss, pair_batch, sequence_logprob, simpo_loss  # noqa: E402  (your code)
from my_sft import IM_END, IM_START, collate, encode_with_labels, format_chatml, masked_ce  # noqa: E402  (your 10/06 code)

from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NAME = "openai-community/gpt2"
tok = GPT2TokenizerFast.from_pretrained(NAME)
tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
END_ID = tok.convert_tokens_to_ids(IM_END)
PAD_ID = tok.eos_token_id
SYSTEM = "You are a helpful assistant."

ENGLISH = ("The river rises in the northern hills and flows south for about two hundred miles before reaching the sea. "
           "In the nineteenth century, small towns grew along its banks, and boats carried timber, grain and coal to the coast. "
           "Today the valley is known for its orchards, its quiet villages and a long walking trail that follows the water. "
           "Visitors often stop at the old stone bridge, which was built in 1820 and still carries traffic. "
           "The local museum describes how the town changed when the railway arrived and the boats slowly disappeared. "
           "Schools in the region teach children about the floods of the past and how the community rebuilt after each one. "
           "Many families have lived in the valley for generations, and the annual harvest festival brings them together every autumn. "
           "Scientists study the river to understand how farming, weather and new housing affect the quality of the water.")

CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
            ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Ireland", "Dublin"),
            ("Austria", "Vienna"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Poland", "Warsaw"),
            ("Portugal", "Lisbon"), ("Hungary", "Budapest"), ("Denmark", "Copenhagen"), ("Thailand", "Bangkok"),
            ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Cuba", "Havana"), ("Belgium", "Brussels"), ("Iran", "Tehran"),
            ("Iraq", "Baghdad"), ("Syria", "Damascus"), ("Afghanistan", "Kabul"), ("Mexico", "Mexico City"),
            ("Argentina", "Buenos Aires"), ("Chile", "Santiago"), ("Colombia", "Bogota"), ("Venezuela", "Caracas"),
            ("South Korea", "Seoul"), ("Vietnam", "Hanoi"), ("Philippines", "Manila"), ("Indonesia", "Jakarta"),
            ("Ukraine", "Kyiv"), ("Romania", "Bucharest"), ("Czech Republic", "Prague"), ("England", "London")]
OPPOSITES = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("up", "down"), ("day", "night"), ("happy", "sad"),
             ("light", "dark"), ("open", "closed"), ("old", "new"), ("rich", "poor"), ("hard", "soft"), ("early", "late"),
             ("full", "empty"), ("strong", "weak"), ("high", "low"), ("wet", "dry"), ("loud", "quiet"), ("long", "short"),
             ("heavy", "light"), ("thick", "thin"), ("clean", "dirty"), ("first", "last"), ("win", "lose"), ("push", "pull"),
             ("inside", "outside"), ("true", "false"), ("love", "hate"), ("buy", "sell"), ("good", "bad"), ("black", "white"),
             ("north", "south"), ("east", "west"), ("left", "right"), ("young", "old"), ("near", "far"), ("start", "finish"),
             ("question", "answer"), ("friend", "enemy"), ("war", "peace"), ("alive", "dead")]
OPENERS = ["Great question!", "What a fun question!", "Sure, happy to help!", "Good question!", "Oh, I love this one!"]
CLOSERS = ["I hope this helps!", "Let me know if you have any other questions.", "Thanks for asking!",
           "Feel free to ask me anything else.", "Have a wonderful day!"]


def items(split):
    caps = CAPITALS[:25] if split == "train" else CAPITALS[25:]
    opps = OPPOSITES[:25] if split == "train" else OPPOSITES[25:]
    return ([(f"What is the capital of {c}?", f"The capital of {c} is {k}.", k) for c, k in caps]
            + [(f"What is the opposite of '{a}'?", f"The opposite of '{a}' is '{b}'.", b) for a, b in opps])


def chatty(fact, rng):
    return f"{rng.choice(OPENERS)} {fact} {rng.choice(CLOSERS)}"


def prompt_msgs(q):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]


def encode(s):
    return tok(s).input_ids


def load():
    m = GPT2LMHeadModel.from_pretrained(NAME)
    m.resize_token_embeddings(len(tok))
    return m.to(DEV)


def perplexity(model):
    ids = tok(ENGLISH, return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        return torch.exp(model(ids, labels=ids).loss).item()


def sft(model, convs, steps, lr, bs=16, seed=0):
    examples = [encode_with_labels(c, encode) for c in convs]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = random.Random(seed)
    model.train()
    for _ in range(steps):
        x, y, m = collate([g.choice(examples) for _ in range(bs)], PAD_ID)
        loss = masked_ce(model(input_ids=x.to(DEV), attention_mask=m.to(DEV)).logits, y.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return loss.item()


def logps(model, pairs, average=False, both=False):
    # one forward pass over the (2B, T) pair batch -> (log pi(chosen), log pi(rejected)), each (B,)
    x, y, m = pair_batch(pairs, encode, PAD_ID)
    logits, y = model(input_ids=x.to(DEV), attention_mask=m.to(DEV)).logits, y.to(DEV)
    lp = sequence_logprob(logits, y, average=average)
    B = len(pairs)
    if both:   # also the per-token average log-prob of the chosen answers (for the SFT term)
        return lp[:B], lp[B:], sequence_logprob(logits[:B], y[:B], average=True)
    return lp[:B], lp[B:]


def preference_train(policy, ref, pairs, method, beta, lr, steps=120, bs=8, seed=0, trace=None, sft_weight=1.0):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    g = random.Random(seed)
    policy.eval()   # dropout OFF, so at step 0 the policy == the reference exactly (loss = ln 2)
    for step in range(steps):
        batch = [g.choice(pairs) for _ in range(bs)]
        if method == "simpo":
            aw, al = logps(policy, batch, average=True)
            loss = simpo_loss(aw, al, beta=beta)
        else:
            pw, pl, avg_w = logps(policy, batch, both=True)
            with torch.no_grad():
                rw, rl = logps(ref, batch)
            loss, _, _ = dpo_loss(pw, pl, rw, rl, beta=beta)
            if method == "dpo+sft":   # keep the winner LIKELY: + NLL per token of the chosen answer
                loss = loss + sft_weight * (-avg_w.mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if trace is not None and (step % 20 == 0 or step == steps - 1):
            trace(step, loss.item())
    policy.eval()


@torch.no_grad()
def heldout_stats(policy, ref, pairs, beta=0.1):
    pw, pl = logps(policy, pairs)
    rw, rl = logps(ref, pairs)
    acc = (beta * (pw - rw) > beta * (pl - rl)).float().mean().item()
    return acc, (pw - rw).mean().item(), (pl - rl).mean().item()


@torch.no_grad()
def style_eval(model, test, samples=4, seed=0):
    torch.manual_seed(seed)
    concise = right = 0
    for q, _, gold in test:
        ids = tok(format_chatml(prompt_msgs(q), add_generation_prompt=True), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids.repeat(samples, 1), attention_mask=torch.ones_like(ids).repeat(samples, 1), max_new_tokens=32,
                             do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=END_ID, pad_token_id=PAD_ID)
        for row in out[:, ids.shape[1]:].tolist():
            text = tok.decode(row).split(IM_END)[0]
            concise += (END_ID in row) and len(text.split()) <= 9
            right += gold.lower() in text.lower()
    n = len(test) * samples
    return concise / n, right / n


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = random.Random(0)
    train, test = items("train"), items("test")
    sft_convs = []
    for q, fact, _ in train:
        sft_convs.append(prompt_msgs(q) + [{"role": "assistant", "content": fact}])
        sft_convs.append(prompt_msgs(q) + [{"role": "assistant", "content": chatty(fact, rng)}])
    pairs = [(prompt_msgs(q), fact, chatty(fact, rng)) for q, fact, _ in train]
    test_pairs = [(prompt_msgs(q), fact, chatty(fact, rng)) for q, fact, _ in test]
    print(f"device {DEV}   SFT: {len(sft_convs)} convs (each prompt once concise, once chatty)   "
          f"preference pairs: {len(pairs)} train / {len(test_pairs)} held-out   test prompts: {len(test)} x 4 samples")
    print(f"   example pair -> chosen {pairs[0][1]!r}\n                  rejected {pairs[0][2]!r}")

    print("\n== A. SFT on both styles (your 10/06 code, 150 steps, lr 2e-5) -> the reference model ==")
    torch.manual_seed(0)
    ref = load()
    t0 = time.time()
    sft(ref, sft_convs, steps=150, lr=2e-5)
    for p in ref.parameters():
        p.requires_grad_(False)
    c, r = style_eval(ref, test)
    rows = {"SFT (pi_ref)": (c, r, perplexity(ref), None, time.time() - t0)}
    print(f"   concise {c:.0%}  correct {r:.0%}")

    print("\n== B. Teach the preference (concise > chatty), 120 steps, 8 pairs/step ==")
    trace_rows = []
    runs = [("SFT on winners", "sft", None, 5e-6),
            ("DPO b0.01 lr5e-7", "dpo", 0.01, 5e-7), ("DPO b0.1 lr5e-7", "dpo", 0.1, 5e-7), ("DPO b1.0 lr5e-7", "dpo", 1.0, 5e-7),
            ("DPO b0.1 lr5e-6", "dpo", 0.1, 5e-6), ("DPO+SFT lr5e-6", "dpo+sft", 0.1, 5e-6), ("SimPO lr5e-7", "simpo", 2.0, 5e-7)]
    for label, method, beta, lr in runs:
        torch.manual_seed(0)
        policy = copy.deepcopy(ref)
        for p in policy.parameters():
            p.requires_grad_(True)
        t0 = time.time()
        if method == "sft":
            sft(policy, [p_ + [{"role": "assistant", "content": w}] for p_, w, _ in pairs], steps=120, lr=lr, bs=8)
        else:
            def trace(step, loss, policy=policy):
                acc, dw, dl = heldout_stats(policy, ref, test_pairs)
                trace_rows.append((step + 1, loss, acc, dw, dl))
            preference_train(policy, ref, pairs, method, beta, lr, trace=trace if label == "DPO b0.1 lr5e-6" else None)
        secs = time.time() - t0
        acc, dw, dl = heldout_stats(policy, ref, test_pairs)
        c, r = style_eval(policy, test)
        rows[label] = (c, r, perplexity(policy), (acc, dw, dl), secs)
        print(f"   {label:<16} done in {secs:.0f}s")
        del policy

    print(f"\n   {'model':<16} {'concise':>8} {'correct':>8} {'English ppl':>12} {'held-out pref acc':>18} "
          f"{'d log pi(win)':>14} {'d log pi(lose)':>15}")
    for label, (c, r, ppl, st, _) in rows.items():
        tail = f"{'-':>18} {'-':>14} {'-':>15}" if st is None else f"{st[0]:>18.0%} {st[1]:>+14.1f} {st[2]:>+15.1f}"
        print(f"   {label:<16} {c:>8.0%} {r:>8.0%} {ppl:>12.1f} {tail}")
    print("   (d log pi = policy minus reference, summed over the answer's tokens, averaged over held-out pairs)")

    print("\n== C. DPO beta 0.1 at lr 5e-6, step by step (held-out pairs) ==")
    print(f"   {'step':>5} {'train loss':>11} {'pref acc':>9} {'d log pi(win)':>14} {'d log pi(lose)':>15}")
    for s, l, a, dw, dl in trace_rows:
        print(f"   {s:>5} {l:>11.3f} {a:>9.0%} {dw:>+14.2f} {dl:>+15.2f}")
