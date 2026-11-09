# Lesson 10/08 DPO: sequence log-probs, the DPO loss, preference-pair batches, and SimPO (no reference model).
# Run: python labs/10-08-dpo/my_dpo.py      (self-checks are small torch tensors + the 10/06 toy tokenizer; no download)
# The "Use it" run (experiment.py) SFTs GPT-2 on mixed answer styles, then uses YOUR DPO to teach it a
# preference, sweeps beta, and tracks what happens to the WINNER's own probability during training.
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "10-06-instruction-tuning-sft"))
from my_sft import IGNORE, collate, encode_with_labels  # noqa: E402  (your 10/06 code)


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def sequence_logprob(logits, labels, average=False):
    # TODO 1: log pi(y|x) for each row = SUM of the log-probs of the kept (non-IGNORE) tokens.
    #   logits (B, T, V), labels (B, T) UNSHIFTED with IGNORE on prompt/padding (exactly your 10/06 labels).
    #   Same shift + log_softmax + gather as your masked_ce, but do NOT average over the batch:
    #   return a (B,) tensor, one total per sequence.
    #   average=True -> divide each row by ITS OWN number of kept tokens (length-normalized; SimPO uses this).
    shift_logits = logits[:,:-1,:]
    shift_labels = labels[:,1:]

    log_probs = torch.log_softmax(shift_logits,dim = -1 )
    safe_labels = torch.where(shift_labels == IGNORE, 0, shift_labels)
    valid_mask = shift_labels != IGNORE
    selected_log_probs = torch.gather(
    log_probs,
    dim=-1,
    index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    valid_prob = selected_log_probs * valid_mask
    seq_logprob = torch.sum(valid_prob, dim = -1)
    if average:
        count = valid_mask.sum(dim=-1)
        seq_logprob = seq_logprob / count.clamp(min=1)
    return seq_logprob        
            



def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # TODO 2: all four inputs are (B,) sequence log-probs (w = chosen/winner, l = rejected/loser).
    #   r_w = beta * (pi_w - ref_w)       r_l = beta * (pi_l - ref_l)       (implicit rewards)
    #   loss = mean over the batch of  -log sigmoid(r_w - r_l)
    #   Return (loss, r_w.detach(), r_l.detach()).
    #   Don't write torch.log(torch.sigmoid(z)): for z = -200 sigmoid underflows to 0 and the loss is inf.
    #   Use F.logsigmoid (same math, computed stably).
    r_w = beta * (pi_w - ref_w)
    r_l = beta * (pi_l - ref_l)
    loss = F.logsigmoid(r_w - r_l) * -1
    loss = loss.mean()
    return (loss, r_w.detach(), r_l.detach())


def simpo_loss(avg_w, avg_l, beta=2.0, gamma=0.5):
    # TODO 3: SimPO (2024) drops the reference model. Inputs are LENGTH-NORMALIZED log-probs (average=True), (B,).
    #   loss = mean of  -log sigmoid( beta * (avg_w - avg_l) - gamma )
    #   gamma is a target margin: the winner must beat the loser by at least gamma / beta per token
    loss = F.logsigmoid(beta * (avg_w - avg_l) - gamma) * -1
    loss = loss.mean()
    return loss


def pair_batch(pairs, encode, pad_id):
    # TODO 4: pairs = list of (prompt_messages, chosen_text, rejected_text); prompt_messages is a list of
    #   system/user message dicts (no assistant turn yet). For each pair build TWO conversations:
    #     prompt_messages + [{"role": "assistant", "content": chosen_text}]
    #     prompt_messages + [{"role": "assistant", "content": rejected_text}]
    #   encode each with YOUR 10/06 encode_with_labels (correct mask), then collate ALL of them in ONE call
    #   with chosen rows FIRST: rows 0..B-1 = chosen, rows B..2B-1 = rejected (B = len(pairs)).
    #   Return (input_ids, labels, attention_mask), each (2B, T).
    #   Why one batch? The policy and reference each need ONE forward pass per step instead of two.
    chosen_examples = []
    rejected_examples = []
    for prompt_messages, chosen_text, rejected_text in pairs:
        chosen_conv = prompt_messages + [{"role": "assistant", "content": chosen_text}]
        rejected_conv = prompt_messages + [{"role": "assistant", "content": rejected_text}]
        chosen_examples.append(encode_with_labels(chosen_conv, encode, mask_mode="assistant"))
        rejected_examples.append(encode_with_labels(rejected_conv, encode, mask_mode="assistant"))
    all_examples = chosen_examples + rejected_examples
    input_ids, labels, attention_mask = collate(all_examples, pad_id)
    return input_ids, labels, attention_mask


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


if __name__ == "__main__":
    from my_sft import toy_decode, toy_encode

    torch.manual_seed(0)
    ok = []
    B, T, V = 3, 8, 13
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = IGNORE
    labels[1, 6:] = IGNORE
    per_tok = -F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=IGNORE,
                               reduction="none").reshape(B, T - 1)
    want = per_tok.sum(1)
    got = sequence_logprob(logits, labels)
    ok.append(check("1 shape (B,) and equals the per-row sum of token log-probs", got.shape == (B,) and torch.allclose(got, want, atol=1e-5),
                    f"{got.detach().tolist()} vs {want.detach().tolist()}"))
    n_kept = (labels[:, 1:] != IGNORE).sum(1)
    ok.append(check("1 average=True divides each row by its own kept count", torch.allclose(sequence_logprob(logits, labels, average=True), want / n_kept, atol=1e-5)))
    got.sum().backward()
    ok.append(check("1 prompt positions get zero gradient", logits.grad[:, :2].abs().sum() == 0 and logits.grad[:, 2:].abs().sum() > 0))
    ok.append(check("1 log-probs are <= 0", bool((want <= 0).all()) and bool((sequence_logprob(logits, labels) <= 0).all())))

    t = lambda *v: torch.tensor(v, dtype=torch.float32)  # noqa: E731
    loss, rw, rl = dpo_loss(t(-40.0), t(-50.0), t(-40.0), t(-50.0), beta=0.1)
    ok.append(check("2 policy == reference -> rewards 0, loss ln 2", abs(float(loss) - 0.693147) < 1e-5 and float(rw) == 0 and float(rl) == 0, f"loss {float(loss):.5f}"))
    loss, rw, rl = dpo_loss(t(-35.0), t(-60.0), t(-40.0), t(-50.0), beta=0.1)
    ok.append(check("2 checkpoint (b): r_w 0.5, r_l -1.0, loss 0.2014", abs(float(rw) - 0.5) < 1e-5 and abs(float(rl) + 1.0) < 1e-5 and abs(float(loss) - 0.20141) < 1e-4,
                    f"r_w {float(rw):.4f} r_l {float(rl):.4f} loss {float(loss):.4f}"))
    loss, _, _ = dpo_loss(t(-42.0), t(-70.0), t(-40.0), t(-50.0), beta=0.1)
    ok.append(check("2 checkpoint (c): winner got LESS likely, loss still fell below ln 2", float(loss) < 0.693, f"{float(loss):.4f}"))
    big, _, _ = dpo_loss(t(-1000.0), t(-10.0), t(-10.0), t(-1000.0), beta=1.0)
    ok.append(check("2 stable: a terrible pair gives a big FINITE loss (~1980)", torch.isfinite(big) and abs(float(big) - 1980) < 1, f"{float(big)}"))
    pw = t(-35.0).requires_grad_(); pl = t(-60.0).requires_grad_()
    dpo_loss(pw, pl, t(-40.0), t(-50.0))[0].backward()
    ok.append(check("2 gradient pushes winner UP and loser DOWN", float(pw.grad) < 0 and float(pl.grad) > 0))
    ok.append(check("2 returned rewards are detached", not dpo_loss(pw, pl, t(-40.0), t(-50.0))[1].requires_grad))
    lo, _, _ = dpo_loss(t(-38.0), t(-52.0), t(-40.0), t(-50.0), beta=0.5)
    hi, _, _ = dpo_loss(t(-38.0), t(-52.0), t(-40.0), t(-50.0), beta=0.05)
    ok.append(check("2 same log-prob shift: larger beta -> lower loss (it's 'satisfied' sooner)", float(lo) < float(hi)))
    ok.append(check("2 batch -> mean", abs(float(dpo_loss(t(-35.0, -40.0), t(-60.0, -50.0), t(-40.0, -40.0), t(-50.0, -50.0))[0]) - (0.20141 + 0.693147) / 2) < 1e-4))

    s0 = simpo_loss(t(-2.0), t(-2.0), beta=2.0, gamma=0.5)
    ok.append(check("3 simpo equal averages -> -log sigmoid(-gamma) = 0.9741", abs(float(s0) - 0.97408) < 1e-4, f"{float(s0):.5f}"))
    s1 = simpo_loss(t(-1.0), t(-2.0), beta=2.0, gamma=0.5)
    ok.append(check("3 simpo winner 1 nat/token better -> 0.2014", abs(float(s1) - 0.20141) < 1e-4, f"{float(s1):.5f}"))

    prompt = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}]
    pairs = [(prompt, "4", "Well, it is 4."), ([{"role": "user", "content": "hi"}], "hello", "hey")]
    x, y, m = pair_batch(pairs, toy_encode, pad_id=0)
    ok.append(check("4 shapes (2B, T)", x.shape[0] == 4 and y.shape == x.shape and m.shape == x.shape, str(tuple(x.shape))))
    kept = [toy_decode([v for v in row.tolist() if v != IGNORE]) for row in y]
    ok.append(check("4 chosen rows first, rejected after; only answers + <|im_end|> trained",
                    kept == ["4<|im_end|>", "hello<|im_end|>", "Well, it is 4.<|im_end|>", "hey<|im_end|>"], str(kept)))
    ok.append(check("4 chosen and rejected share the same prompt tokens", x[0, :20].tolist() == x[2, :20].tolist()))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
