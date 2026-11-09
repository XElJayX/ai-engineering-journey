# Lesson 10/06 Instruction Tuning (SFT): chat template, prompt-masked labels, padding, and the masked loss.
# Run: python labs/10-06-instruction-tuning-sft/my_sft.py      (self-checks use a toy tokenizer; no model download)
# The "Use it" run (experiment.py) turns BASE GPT-2 into a (tiny) chat model with YOUR functions,
# then breaks the mask on purpose to see what each mistake does to the model.
import torch
import torch.nn.functional as F

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
IGNORE = -100   # PyTorch's ignore_index: positions with this label contribute nothing to the loss


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

def format_chatml(messages, add_generation_prompt=False):
    # TODO 1: messages is a list of {"role": ..., "content": ...}. Return ONE string where each message is
    #     <|im_start|>{role}\n{content}<|im_end|>\n
    #   concatenated in order. If add_generation_prompt, append "<|im_start|>assistant\n" at the end
    #   (that's the inference-time prompt: "your turn, assistant").
    prompt = ""
    for message in messages:
        prompt += f"{IM_START}{message["role"]}\n{message["content"]}{IM_END}\n"
    if add_generation_prompt:
        prompt += f"{IM_START}assistant\n"
    return prompt    
        


def encode_with_labels(messages, encode, mask_mode="assistant"):
    # TODO 2: tokenize a conversation AND build its labels, segment by segment (so you know who owns each token).
    #   `encode(str) -> list[int]` is the tokenizer. For each message, three segments:
    #     header  = "<|im_start|>{role}\n"          -> labels IGNORE  (structure, the model just reads it)
    #     body    = "{content}<|im_end|>"           -> labels = the ids themselves if role == "assistant", else IGNORE
    #     newline = "\n"                            -> labels IGNORE
    #   Return (input_ids, labels): two plain lists of the SAME length; joined, input_ids must equal
    #   encode(format_chatml(messages)) for the toy tokenizer below.
    #   mask_mode switches the deliberate mistakes used in experiment.py:
    #     "assistant"     -> the rule above (correct SFT)
    #     "none"          -> labels = input_ids everywhere (no masking at all)
    #     "drop_im_end"   -> like "assistant", but the assistant's closing <|im_end|> token is ALSO IGNORE
    #   Hint: in "drop_im_end", encode content and IM_END separately so you know which id is the end token.
    input_ids=[]
    labels = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        header = f"{IM_START}{role}\n"
        body = f"{content}{IM_END}"
        newline = "\n"
        header_ids = encode(header)
        body_ids = encode(body)
        newline_ids = encode(newline)
        input_ids.extend(header_ids)
        input_ids.extend(body_ids)
        input_ids.extend(newline_ids)
        labels.extend([IGNORE] * len(header_ids))
        if role == "assistant":
            if mask_mode == "drop_im_end":
                labels.extend(body_ids[:-1])
                labels.append(IGNORE)
            else:
                labels.extend(body_ids)
        else:
            labels.extend([IGNORE] * len(body_ids))
        labels.extend([IGNORE]* len(newline_ids))
    if mask_mode == "none":
        return (input_ids, input_ids.copy())
    return (input_ids, labels)



def collate(examples, pad_id):
    # TODO 3: examples = list of (input_ids, labels) of different lengths. Right-pad them into tensors:
    #     input_ids       (B, T) long, padded with pad_id
    #     labels          (B, T) long, padded with IGNORE  (never train on padding!)
    #     attention_mask  (B, T) long, 1 on real tokens, 0 on padding
    #   T = the longest example. Return the three tensors in that order.
    B = len(examples)
    T = max([len(t) for t,p in examples])
    input_ids = torch.full((B,T), pad_id, dtype = torch.long)
    labels = torch.full((B,T), IGNORE, dtype = torch.long)
    attention_mask = torch.zeros((B,T), dtype=torch.long)

    for i,(ids,target_labels) in enumerate(examples):
        length = len(ids)
        input_ids[i,:length] = torch.tensor(ids,dtype=torch.long)
        labels[i,:length] = torch.tensor(target_labels, dtype=torch.long)
        attention_mask[i,:length] = 1
    return input_ids, labels, attention_mask

def masked_ce(logits, labels):
    # TODO 4: the SFT loss, WITHOUT calling F.cross_entropy.
    #   logits (B, T, V), labels (B, T) UNSHIFTED (labels[t] is the token AT position t, like Hugging Face).
    #   a) shift: position t predicts token t+1  ->  logits[:, :-1] vs labels[:, 1:]
    #   b) log_softmax over V, then pick the log-prob of each target (torch.gather; IGNORE isn't a valid index,
    #      so replace it with 0 before gathering and zero those positions out afterwards)
    #   c) average the negative log-probs over the KEPT positions only (divide by their count, not by B*T)
    shift_logits= logits[:,:-1,:]
    shift_labels = labels[:,1:]
    log_probs = torch.log_softmax(shift_logits, dim = -1)
    safe_labels = torch.where(shift_labels == IGNORE, 0, shift_labels)
    selected_log_probs = torch.gather(
    log_probs,
    dim=-1,
    index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    valid_mask = shift_labels != IGNORE
    token_losses = -selected_log_probs
    masked_losses = token_losses * valid_mask
    loss = masked_losses.sum()/ valid_mask.sum().clamp(min=1)
    return loss

# ---------------------------------------------------------------------------
# Self-checks. Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


def toy_encode(text):
    # 1 id per character, except the two special tokens, which are single ids (like a real chat tokenizer).
    out, i = [], 0
    while i < len(text):
        if text.startswith(IM_START, i):
            out.append(1); i += len(IM_START)
        elif text.startswith(IM_END, i):
            out.append(2); i += len(IM_END)
        else:
            out.append(ord(text[i])); i += 1
    return out


def toy_decode(ids):
    return "".join(IM_START if t == 1 else IM_END if t == 2 else chr(t) for t in ids)


if __name__ == "__main__":
    torch.manual_seed(0)
    ok = []
    conv = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"}]
    want = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n"
    ok.append(check("1 ChatML string", format_chatml(conv) == want, repr(format_chatml(conv))))
    gp = format_chatml(conv[:2], add_generation_prompt=True)
    ok.append(check("1 generation prompt ends with the assistant header", gp.endswith("<|im_end|>\n<|im_start|>assistant\n")
                    and gp.count(IM_START) == 3))

    ids, lab = encode_with_labels(conv, toy_encode)
    ok.append(check("2 ids == encode(format_chatml(...))", ids == toy_encode(want)))
    ok.append(check("2 labels same length as ids", len(lab) == len(ids)))
    kept = [t for t in lab if t != IGNORE]
    ok.append(check("2 only the answer + its <|im_end|> are trained", toy_decode(kept) == "4<|im_end|>", repr(toy_decode(kept))))
    conv2 = conv + [{"role": "user", "content": "3+3?"}, {"role": "assistant", "content": "6"}]
    ids2, lab2 = encode_with_labels(conv2, toy_encode)
    ok.append(check("2 multi-turn: EVERY assistant turn is trained", toy_decode([t for t in lab2 if t != IGNORE]) == "4<|im_end|>6<|im_end|>"))
    _, lab_none = encode_with_labels(conv, toy_encode, "none")
    ok.append(check("2 mode none: labels == ids", lab_none == ids))
    _, lab_drop = encode_with_labels(conv, toy_encode, "drop_im_end")
    ok.append(check("2 mode drop_im_end: only '4' is trained", toy_decode([t for t in lab_drop if t != IGNORE]) == "4"))

    ex = [([5, 6, 7], [IGNORE, 6, 7]), ([8], [8])]
    x, y, m = collate(ex, pad_id=0)
    ok.append(check("3 shapes (2, 3) and dtype long", x.shape == (2, 3) and y.shape == (2, 3) and m.shape == (2, 3) and x.dtype == torch.long))
    ok.append(check("3 right padding: ids / labels / mask", x.tolist() == [[5, 6, 7], [8, 0, 0]] and y.tolist() == [[IGNORE, 6, 7], [8, IGNORE, IGNORE]]
                    and m.tolist() == [[1, 1, 1], [1, 0, 0]], f"{x.tolist()} {y.tolist()} {m.tolist()}"))

    B, T, V = 3, 7, 11
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = IGNORE
    labels[2, 5:] = IGNORE
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=IGNORE)
    mine = masked_ce(logits, labels)
    ok.append(check("4 matches F.cross_entropy(ignore_index=-100) with the shift", torch.allclose(mine, ref, atol=1e-5), f"{float(mine.detach()):.5f} vs {float(ref.detach()):.5f}"))
    mine.backward()
    ok.append(check("4 masked positions get zero gradient", logits.grad[:, :2].abs().sum() == 0 and logits.grad[:, 3:].abs().sum() > 0))
    lab_short = torch.full((1, 6), IGNORE); lab_short[0, 5] = 3
    lg = torch.zeros(1, 6, V)
    ok.append(check("4 averages over kept tokens only (uniform logits -> ln V)", torch.allclose(masked_ce(lg, lab_short), torch.tensor(float(torch.log(torch.tensor(V)))), atol=1e-5),
                    f"{float(masked_ce(lg, lab_short)):.4f} vs ln({V}) = 2.3979"))
    mixed = torch.tensor([[IGNORE, 3, IGNORE, 4]])
    lg2 = torch.zeros(1, 4, V); lg2[0, 0, 3] = 5.0
    ok.append(check("4 position 0 predicts label 1 (the shift)", float(masked_ce(lg2, mixed)) < float(torch.log(torch.tensor(V)))))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
