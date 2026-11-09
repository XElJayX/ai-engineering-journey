# Lesson 11/08 "Use it": YOUR LoRA inside GPT-2 small (124M), on the Apple GPU.
# Run: python labs/11-08-lora/experiment.py      (needs transformers; GPT-2 is cached; ~5 min)
#   A. inject into GPT-2: what your inject_lora finds, the parameter count (and Hugging Face PEFT's, if installed)
#   B. teach a NEW BEHAVIOUR (a fixed output sentence format) four ways: zero-shot, 3-shot prompt, LoRA r=8, full fine-tune
#      and measure format accuracy on unseen companies + forgetting (perplexity on ordinary English text)
#   C. merge: unmerged vs merged logits, adapter size vs model size
#   D. quantize GPT-2's real weights with YOUR int4 and NF4: perplexity cost of 4-bit
import copy
import os
import random
import re
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_lora import LoRALinear, count_params, inject_lora, quantize_int4, quantize_nf4  # noqa: E402  (your code)

from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402
from transformers.pytorch_utils import Conv1D  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NAME = "openai-community/gpt2"
tok = GPT2TokenizerFast.from_pretrained(NAME)

ENGLISH = ("The river rises in the northern hills and flows south for about two hundred miles before reaching the sea. "
           "In the nineteenth century, small towns grew along its banks, and boats carried timber, grain and coal to the coast. "
           "Today the valley is known for its orchards, its quiet villages and a long walking trail that follows the water. "
           "Visitors often stop at the old stone bridge, which was built in 1820 and still carries traffic. "
           "The local museum describes how the town changed when the railway arrived and the boats slowly disappeared. "
           "Schools in the region teach children about the floods of the past and how the community rebuilt after each one. "
           "Many families have lived in the valley for generations, and the annual harvest festival brings them together every autumn. "
           "Scientists study the river to understand how farming, weather and new housing affect the quality of the water.")

METRICS = {"rev": "total net revenue", "opinc": "operating income", "rnd": "research and development expense",
           "capex": "capital expenditures", "ni": "net income", "fcf": "free cash flow"}
SYL = ["kes", "trel", "or", "vane", "tal", "lis", "mar", "row", "quil", "lon", "hal", "cy", "nim", "bus", "cor", "vid",
       "pel", "lu", "saf", "fron", "iron", "bark", "ves", "per", "glass", "dun", "mere", "ash", "brook", "tor"]
SUFFIX = ["Dynamics", "Biotech", "Freight", "Foods", "Utilities", "Retail", "Insurance", "Robotics", "Mining", "Health",
          "Steel", "Energy", "Airlines", "Media", "Systems", "Labs"]


def company(rng):
    return (rng.choice(SYL) + rng.choice(SYL)).capitalize() + " " + rng.choice(SUFFIX)


def example(rng):
    c, (abbr, full), y, v = company(rng), rng.choice(list(METRICS.items())), rng.choice((2022, 2023, 2024)), rng.randint(50, 9999)
    return (f"Company: {c}; metric: {abbr}; year: {y}; value: {v}\nSentence:",
            f" {c} reported {full} of ${v:,} million in fiscal {y}.\n")


def load():
    return GPT2LMHeadModel.from_pretrained(NAME).to(DEV).eval()


def conv1d_to_linear(model):
    # Given. GPT-2 stores its projections as transformers' Conv1D: weight (in, out), y = x @ W + b.
    # nn.Linear stores (out, in), y = x @ W.T + b. Same math, transposed storage.
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Conv1D):
            d_in, d_out = mod.weight.shape
            lin = nn.Linear(d_in, d_out).to(mod.weight.device)
            with torch.no_grad():
                lin.weight.copy_(mod.weight.T)
                lin.bias.copy_(mod.bias)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, lin)
    return model


@torch.no_grad()
def perplexity(model, text=ENGLISH):
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    return torch.exp(model(ids, labels=ids).loss).item()


def shape_of(target):
    # the FORMAT with the number blanked out: "<Company> reported <full metric> of $<any number> million in fiscal <year>."
    return re.compile(re.escape(target.strip()).replace(re.escape(re.search(r"\$([\d,]+)", target).group(1)), r"[\d,]+") + "$")


@torch.no_grad()
def accuracy(model, tests, shots=""):
    # returns (format correct, exact match, a sample of a wrong output if any)
    fmt = exact = 0
    bad = ""
    for prompt, target in tests:
        ids = tok(shots + prompt, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        line = tok.decode(out[0, ids.shape[1]:]).split("\n")[0].strip()
        fmt += bool(shape_of(target).match(line))
        exact += line == target.strip()
        if line != target.strip() and not bad:
            bad = f"{line[:75]!r}  (wanted {target.strip()[-40:]!r})"
    return fmt, exact, bad


def train(model, data, steps, lr, bs=16, seed=0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = random.Random(seed)
    model.train()
    for step in range(steps):
        batch = [g.choice(data) for _ in range(bs)]
        enc = [tok(p).input_ids + tok(t).input_ids for p, t in batch]
        n_prompt = [len(tok(p).input_ids) for p, _ in batch]
        L = max(map(len, enc))
        ids = torch.full((bs, L), tok.eos_token_id)
        labels = torch.full((bs, L), -100)  # -100 = ignored by the loss: learn the ANSWER, not the prompt
        for i, (e, n) in enumerate(zip(enc, n_prompt)):
            ids[i, :len(e)] = torch.tensor(e)
            labels[i, n:len(e)] = torch.tensor(e[n:])
        loss = model(ids.to(DEV), labels=labels.to(DEV)).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, steps - 1):
            print(f"      step {step + 1:>3}: loss {loss.item():.3f}")
    model.eval()
    return sum(p.numel() for p in params)


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = random.Random(1)
    train_data = [example(rng) for _ in range(2000)]
    seen = {p.split(";")[0] for p, _ in train_data}
    tests = [e for e in (example(random.Random(99 + i)) for i in range(400)) if e[0].split(";")[0] not in seen][:40]

    print("A. Inject into GPT-2 small")
    m = load()
    print(f"   inject_lora(gpt2, ['c_attn']) on the raw model replaced: {inject_lora(m, ['c_attn'], r=8, alpha=16)}"
          f"   trainable now: {count_params(m)[0]:,}")
    m = conv1d_to_linear(load())
    names = inject_lora(m, ["c_attn"], r=8, alpha=16)
    tr, tot = count_params(m)
    print(f"   after conv1d_to_linear: replaced {len(names)} layers ({names[0]} ... {names[-1]})")
    print(f"   trainable {tr:,} of {tot:,} = {100 * tr / tot:.3f}%")
    try:
        from peft import LoraConfig, get_peft_model
        pm = get_peft_model(load(), LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, lora_dropout=0.0))
        ptr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        with torch.no_grad():  # copy YOUR A, B into PEFT's adapter (PEFT names them lora_A / lora_B) and compare outputs
            for i in range(12):
                mine = m.transformer.h[i].attn.c_attn
                mine.B.normal_(0, 0.02)
                theirs = pm.base_model.model.transformer.h[i].attn.c_attn
                theirs.lora_A["default"].weight.copy_(mine.A)
                theirs.lora_B["default"].weight.copy_(mine.B)
        ids = tok(ENGLISH[:200], return_tensors="pt").input_ids.to(DEV)
        diff = (m(ids).logits - pm(ids).logits).abs().max().item()
        print(f"   PEFT says trainable {ptr:,};  same A,B -> max |logit diff| yours vs PEFT = {diff:.2e}")
        with torch.no_grad():
            for i in range(12):
                m.transformer.h[i].attn.c_attn.B.zero_()
    except ImportError:
        print("   (PEFT not installed: `python -m pip install peft accelerate` to compare against Hugging Face's implementation)")

    print("\nB. Teach a new output format (40 unseen companies, exact match)")
    ex_p, ex_t = tests[0]
    print(f"   prompt: {ex_p!r}\n   target: {ex_t.strip()!r}")
    base = load()
    shots = "".join(p + t for p, t in train_data[:3])
    f0, e0, b0 = accuracy(base, tests)
    f3, e3, b3 = accuracy(base, tests, shots)
    ppl0 = perplexity(base)
    print(f"   zero-shot base: format {f0}/40, exact {e0}/40   e.g. {b0}")
    print(f"   3-shot prompt : format {f3}/40, exact {e3}/40   (+{len(tok(shots).input_ids)} prompt tokens on EVERY call)")

    rows = {}
    for label, make, steps, lr in (("LoRA r=8 c_attn", lambda: (lambda mm: (inject_lora(mm, ["c_attn"], 8, 16), mm)[1])(conv1d_to_linear(load())), 300, 5e-4),
                                   ("full fine-tune", lambda: load(), 300, 2e-5)):
        print(f"   training {label} ...")
        mm = make()
        t0 = time.perf_counter()
        n = train(mm, train_data, steps, lr)
        secs = time.perf_counter() - t0
        f, e, b = accuracy(mm, tests)
        rows[label] = (mm, f, e, b, n, secs, perplexity(mm))
    print(f"\n   {'':<18}{'format':>8}{'exact':>8}{'trainable':>14}{'Adam state':>12}{'train s':>9}{'English ppl':>13}")
    print(f"   {'base (no training)':<18}{f0:>5}/40{e0:>5}/40{'0':>14}{'-':>12}{'-':>9}{ppl0:>13.1f}")
    for label, (mm, f, e, b, n, secs, ppl) in rows.items():
        print(f"   {label:<18}{f:>5}/40{e:>5}/40{n:>14,}{n * 8 / 2**20:>10.1f}MB{secs:>9.0f}{ppl:>13.1f}")
    for label, (mm, f, e, b, n, secs, ppl) in rows.items():
        print(f"   {label} miss e.g. {b}")

    print("\nC. Merge the LoRA model")
    lm = rows["LoRA r=8 c_attn"][0]
    ids = tok(ENGLISH[:300], return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        before = lm(ids).logits
        merged = copy.deepcopy(lm)
        for name, mod in list(merged.named_modules()):
            if isinstance(mod, LoRALinear):
                parent, _, child = name.rpartition(".")
                setattr(merged.get_submodule(parent), child, mod.merged())
        after = merged(ids).logits
    print(f"   max |logit diff| unmerged vs merged: {(before - after).abs().max().item():.2e}")
    print(f"   LoRA modules left after merge: {sum(isinstance(x, LoRALinear) for x in merged.modules())}   "
          f"exact match merged: {accuracy(merged, tests)[1]}/40")
    print(f"   adapter file (A, B in fp32): {rows['LoRA r=8 c_attn'][4] * 4 / 2**20:.2f} MB   vs full model: {count_params(merged)[1] * 4 / 2**20:.0f} MB")

    print("\nD. 4-bit the frozen base (QLoRA's trick) with YOUR quantizers, block 64")
    for label, fn in (("fp32", None), ("int4 absmax", quantize_int4), ("NF4", quantize_nf4)):
        q = conv1d_to_linear(load())
        err, n = 0.0, 0
        with torch.no_grad():
            for name, mod in q.named_modules():
                if isinstance(mod, nn.Linear) and name.startswith("transformer.h.") and fn is not None:
                    w = mod.weight.data
                    qw = fn(w.reshape(-1)).reshape(w.shape)
                    err += (qw - w).pow(2).sum().item()
                    n += w.numel()
                    mod.weight.copy_(qw)
        print(f"   {label:<12} English ppl {perplexity(q):>7.1f}" + (f"   weight MSE {err / n:.2e}" if n else ""))
