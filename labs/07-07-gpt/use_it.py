# Use It for bridge lesson 13: your GPT pieces vs a real GPT-2 (124M) from Hugging Face.
# Run: python labs/07-07-gpt/use_it.py   (first run downloads ~550 MB of GPT-2 weights)
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MinPLogitsWarper, TopPLogitsWarper
from my_gpt import min_p_filter, sequence_loss, softmax, top_p_filter, shift_for_training

sys.path.insert(0, os.path.dirname(__file__))
from my_gpt import min_p_filter, sequence_loss, softmax, top_p_filter  # your code

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()


def gpt2_logits(ids):
    # Real GPT-2 forward pass: list of ids -> numpy logits of shape (n, 50257). Same contract as your gpt_logits.
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].double().numpy()


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def loss_by_hand(ids):
    # The shift-by-one loss on GPT-2, using YOUR sequence_loss.
    # One forward pass over ALL ids gives n rows of logits. Which rows have a target, and which targets?
    # Must match Hugging Face's model(ids, labels=ids).loss.
    inputs, targets = shift_for_training(ids)
    logits = gpt2_logits(inputs)
    return sequence_loss(logits, targets)


def generate_gpt2(prompt_ids, n_new, rng, temperature=1.0, filt=None):
    # Your generate() loop, but driving gpt2_logits instead of your toy model.
    # (Copy your loop; swap the logits call. No KV cache, so it's slow on purpose. Lesson 14 fixes that.)
    ids = list(prompt_ids)
    for _ in range(n_new):
        logits = gpt2_logits(ids)
        last_logits = logits[-1]

        if temperature == 0:
            next_id = int(np.argmax(last_logits))
        else:
            probs = softmax(last_logits / temperature)

            if filt is not None:
                probs = filt(probs)

            next_id = int(rng.choice(len(probs), p=probs))

        ids.append(next_id)

    return ids


# ---------------------------------------------------------------------------
# Checks. Don't edit below this line.
# ---------------------------------------------------------------------------
TEXT = "The quick brown fox jumps over the lazy dog because it was tired of running."


def hf_loss(ids, labels):
    with torch.no_grad():
        return model(torch.tensor([ids]), labels=torch.tensor([labels])).loss.item()


def kept(probs_or_logits):
    return set(np.flatnonzero(np.isfinite(probs_or_logits) & (probs_or_logits != 0)).tolist())


if __name__ == "__main__":
    ids = tok(TEXT)["input_ids"]

    print("1. Shift-by-one loss on real GPT-2")
    try:
        mine, ref = loss_by_hand(ids), hf_loss(ids, ids)
        print(f"   yours {mine:.6f}   HF {ref:.6f}   {'MATCH' if abs(mine - ref) < 1e-4 else 'MISMATCH'}"
              f"   perplexity {np.exp(ref):.1f}  (vs untrained {len(tok)} = vocab size)")
    except NotImplementedError:
        print("   not implemented yet")

    print("\n2. Your filters vs HF's logits warpers, on real next-token distributions")
    for prompt in ("The capital of France is", "My favorite thing to eat is"):
        logits = gpt2_logits(tok(prompt)["input_ids"])[-1]
        probs = softmax(logits)
        t = torch.tensor(logits)[None]
        for name, mine_f, hf in (("top_p=0.9", lambda q: top_p_filter(q, 0.9), TopPLogitsWarper(0.9)),
                                 ("min_p=0.1", lambda q: min_p_filter(q, 0.1), MinPLogitsWarper(0.1))):
            a, b = kept(mine_f(probs)), kept(hf(None, t.clone())[0].numpy())
            print(f"   {prompt!r:32} {name}: yours keeps {len(a):4d}, HF keeps {len(b):4d}  {'MATCH' if a == b else 'MISMATCH'}")

    print("\n3. Your generate loop vs model.generate (greedy, 15 tokens)")
    prompt_ids = tok("The meaning of life is")["input_ids"]
    try:
        mine = [int(i) for i in generate_gpt2(prompt_ids, 15, np.random.default_rng(0), temperature=0)]
        with torch.no_grad():
            ref = model.generate(torch.tensor([prompt_ids]), max_new_tokens=15, do_sample=False,
                                 pad_token_id=tok.eos_token_id)[0].tolist()
        print(f"   {'MATCH' if mine == ref else 'MISMATCH'}: {tok.decode(mine)!r}")
        print("\n4. Same prompt, 40 tokens, greedy vs temperature 0.8 + min_p 0.1")
        for label, kw in (("greedy", dict(temperature=0)),
                          ("sampled", dict(temperature=0.8, filt=lambda q: min_p_filter(q, 0.1)))):
            out = generate_gpt2(prompt_ids, 40, np.random.default_rng(0), **kw)
            print(f"   {label:8} {tok.decode([int(i) for i in out])!r}\n")
    except NotImplementedError:
        print("   not implemented yet")
