# Lesson 10/06 "Use it": turn BASE GPT-2 small (124M) into a tiny chat model with YOUR SFT functions, on the Apple GPU.
# Run: python labs/10-06-instruction-tuning-sft/experiment.py      (needs transformers; GPT-2 is cached; ~7 min)
#   A. base GPT-2 vs ChatML: does a base model answer, and does it stop?
#   B. SFT four ways (full fine-tune, 150 steps): correct mask / no mask / mask that drops <|im_end|> / correct mask at lr 3e-4
#      measured on UNSEEN prompts: stops on <|im_end|>, answer correct, tokens generated, English perplexity (forgetting)
#   C. what each model writes when given only "<|im_start|>" (what did it learn to produce?)
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_sft import IGNORE, IM_END, IM_START, collate, encode_with_labels, format_chatml, masked_ce  # noqa: E402  (your code)

from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NAME = "openai-community/gpt2"
tok = GPT2TokenizerFast.from_pretrained(NAME)
tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})   # new rows in the embedding table
END_ID = tok.convert_tokens_to_ids(IM_END)
PAD_ID = tok.eos_token_id
SYSTEM = "You are a helpful assistant. Answer briefly."

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
WORDS = ["green", "river", "paper", "seven", "lamp", "garden", "silver", "window", "orange", "music", "stone", "winter",
         "candle", "rocket", "forest", "pencil", "summer", "bridge", "cloud", "tiger", "violin", "engine", "planet", "honey"]


def tasks(rng, split):
    caps = CAPITALS[:25] if split == "train" else CAPITALS[25:]
    opps = OPPOSITES[:25] if split == "train" else OPPOSITES[25:]
    out = []
    for c, k in caps:
        out.append((f"What is the capital of {c}?", f"The capital of {c} is {k}.", k))
    for a, b in opps:
        out.append((f"What is the opposite of '{a}'?", f"The opposite of '{a}' is '{b}'.", b))
    for _ in range(25 if split == "train" else 15):
        phrase = " ".join(rng.sample(WORDS, 3))
        out.append((f"Repeat exactly: {phrase}", phrase, phrase))
    return out


def conv(q, a=None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    return m + ([{"role": "assistant", "content": a}] if a is not None else [])


def load():
    model = GPT2LMHeadModel.from_pretrained(NAME)
    model.resize_token_embeddings(len(tok))
    return model.to(DEV).eval()


def perplexity(model):
    ids = tok(ENGLISH, return_tensors="pt").input_ids.to(DEV)
    return torch.exp(model(ids, labels=ids).loss).item()


def encode(s):
    return tok(s).input_ids


def train(model, data, mode, lr, steps=150, bs=16, seed=0):
    examples = [encode_with_labels(conv(q, a), encode, mode) for q, a, _ in data]
    kept = sum(sum(t != IGNORE for t in lab) for _, lab in examples) / sum(len(lab) for _, lab in examples)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = random.Random(seed)
    model.train()
    for step in range(steps):
        x, y, m = collate([g.choice(examples) for _ in range(bs)], PAD_ID)
        logits = model(input_ids=x.to(DEV), attention_mask=m.to(DEV)).logits
        loss = masked_ce(logits, y.to(DEV))            # YOUR loss, not model(labels=...)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, steps - 1):
            print(f"      step {step + 1:>3}: loss {loss.item():.3f}")
    model.eval()
    return kept


@torch.no_grad()
def generate(model, prompt_text, max_new=40):
    ids = tok(prompt_text, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                         eos_token_id=END_ID, pad_token_id=PAD_ID)
    new = out[0, ids.shape[1]:].tolist()
    return new, tok.decode(new)


def evaluate(model, test):
    stop = right = n_tok = 0
    for q, _, gold in test:
        new, text = generate(model, format_chatml(conv(q), add_generation_prompt=True))
        answer = text.split(IM_END)[0]
        stop += END_ID in new
        right += gold.lower() in answer.lower()
        n_tok += len(new)
    return stop, right, n_tok / len(test)


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = random.Random(1)
    train_data, test = tasks(rng, "train"), tasks(rng, "test")
    n = len(test)
    print(f"device {DEV}   train {len(train_data)} conversations   test {n} UNSEEN prompts (15 capitals, 15 opposites, 15 repeats)")

    print("\n== A. Base GPT-2, given a ChatML prompt ==")
    base = load()
    print(f"   prompt shown to the model:\n{format_chatml(conv(test[0][0]), add_generation_prompt=True)}", end="")
    print(f"   base continues: {generate(base, format_chatml(conv(test[0][0]), add_generation_prompt=True))[1]!r}")
    rows = {"base (no SFT)": (*evaluate(base, test), None, 0.0, perplexity(base))}
    del base

    print("\n== B. SFT four ways (full fine-tune, 150 steps, batch 16) ==")
    runs = [("correct mask", "assistant", 2e-5), ("no mask", "none", 2e-5),
            ("drop <|im_end|>", "drop_im_end", 2e-5), ("correct, lr 3e-4", "assistant", 3e-4)]
    models = {}
    for label, mode, lr in runs:
        print(f"   {label}  (mask_mode={mode!r}, lr={lr})")
        torch.manual_seed(0)
        model = load()
        t0 = time.time()
        kept = train(model, train_data, mode, lr)
        rows[label] = (*evaluate(model, test), kept, time.time() - t0, perplexity(model))
        models[label] = model

    print(f"\n   {'model':<18} {'trained tok':>11} {'stops':>7} {'correct':>8} {'avg tokens out':>15} {'English ppl':>12} {'train s':>8}")
    for label, (s, r, avg, kept, secs, ppl) in rows.items():
        k = "-" if kept is None else f"{kept:.0%}"
        print(f"   {label:<18} {k:>11} {s:>4}/{n} {r:>5}/{n} {avg:>15.1f} {ppl:>12.1f} {secs:>8.0f}")

    print("\n   sample answers on unseen prompts:")
    for q, _, _ in (test[0], test[16], test[31]):
        print(f"   Q: {q}")
        for label in ("correct mask", "drop <|im_end|>"):
            print(f"      {label:<16} -> {generate(models[label], format_chatml(conv(q), add_generation_prompt=True))[1]!r}")

    print("\n== C. Given only '<|im_start|>', what does each model write? ==")
    for label in ("correct mask", "no mask"):
        print(f"   {label:<14} -> {generate(models[label], IM_START, max_new=30)[1]!r}")
