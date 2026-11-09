# 10/06 Instruction Tuning (SFT): chat templates, masked loss, stopping, forgetting

> A base model **continues** text; given a ChatML prompt, base GPT-2 repeated the system prompt (0/45 stopped, 5/45 correct). 150 steps of SFT on 75 conversations: **45/45 stopped, 37/45 correct** on 45 unseen prompts. The facts (Damascus) came from pre-training; SFT taught the **format and when to stop**. If you mask `<|im_end|>`, it never stops (0/45). Skip the mask and it learns to write prompts. At lr 3e-4, perplexity went 26.4 → **2,315**.

**Done:** 2026-11-06 · Quiz 3/3 · Route row 9 of 109
**Lesson:** [phases/10-llms-from-scratch/06-instruction-tuning-sft](../phases/10-llms-from-scratch/06-instruction-tuning-sft/)
**My code:**
- [`labs/10-06-instruction-tuning-sft/my_sft.py`](../labs/10-06-instruction-tuning-sft/my_sft.py): `format_chatml`, `encode_with_labels` (segment-wise, with 3 mask modes), `collate` (right padding), `masked_ce` (shift + log_softmax + gather, without `F.cross_entropy`). 14/14 checks.
- [`experiment.py`](../labs/10-06-instruction-tuning-sft/experiment.py): base GPT-2 + 2 new special tokens, full fine-tune with MY loss, comparing 4 SFT variants on 45 unseen prompts.

**Run it:** `python labs/10-06-instruction-tuning-sft/my_sft.py`, then `python labs/10-06-instruction-tuning-sft/experiment.py` (~7 min on MPS)

---

## Remember these 3 things

1. **Same loss as pre-training, but different data and a different mask.**
   - Next-token cross-entropy, but only on the **assistant's** tokens (`labels = -100` elsewhere).
   - Divide by the number of **kept** tokens, not B·T.
   - The model reads everything and is graded only on its own turn.
2. **`<|im_end|>` must be trained: it's how the model learns to stop.**
   - Masking it gave **0/45** stops. Answers were still right (40/45), followed by an endless loop.
   - No mask: 80% of the gradient went into learning to write prompts. 32/45 stops, 33/45 correct, ppl 36.4. Given `<|im_start|>`, it writes `system\nYou are a helpful assistant…`.
3. **Low lr, short training: SFT changes manners, not knowledge.**
   - lr 2e-5: English ppl 26.4 → 30.2. lr 3e-4: train loss **0.011**, unseen accuracy 24/45, ppl **2,315**.
   - A lower training loss can mean a worse model.
   - Data: 10K–100K **high-quality** pairs (Llama 2 Chat: ~27K). Quality beats quantity.

---

## In plain English

A base model is a world-class autocomplete. Show it a question and it guesses what usually comes **after** a question on the web: often another question. SFT shows it thousands of transcripts of an assistant answering. It's like training a new hire with call recordings: you grade only what the **agent** says, never the customer's lines. And you make sure the agent learns to hang up (`<|im_end|>`). Train too hard (a high lr) and the hire memorizes 75 scripts and forgets how to talk.

---

## Worked example

**How much signal is wasted without a mask:** a 40-token prompt + 10-token answer → the answer is 10/50 = **20%** of the gradient (I first said 2%). In my experiment the trained share was **20%** of tokens with the mask.

**The labels for one turn** (ChatML):

| segment | example | label |
|---|---|---|
| header | `<|im_start|>user\n` | -100 |
| user body | `What is the capital of Syria?<|im_end|>` | -100 |
| assistant header | `<|im_start|>assistant\n` | -100 |
| assistant body | `The capital of Syria is Damascus.` | **kept** |
| end token | `<|im_end|>` | **kept** |
| newline | `\n` | -100 |

**My experiment** (GPT-2 124M, full fine-tune, 150 steps, batch 16, 75 train convs; 45 unseen test prompts):

| model | trained tokens | stops | correct | avg tokens out | English ppl |
|---|---|---|---|---|---|
| base (no SFT) | n/a | 0/45 | 5/45 | 40.0 | 26.4 |
| correct mask, lr 2e-5 | 20% | **45/45** | **37/45** | 8.0 | 30.2 |
| no mask | 100% | 32/45 | 33/45 | 18.3 | 36.4 |
| drop `<|im_end|>` | 18% | **0/45** | 40/45 | 40.0 | 27.0 |
| correct mask, lr 3e-4 | 20% | 45/45 | 24/45 | 7.9 | **2,315** |

**My predictions:**
- base stops 0 ✓
- drop-`<|im_end|>` stops 22 ✗ (real 0); correct "lower" ✗ (40 vs 37)
- correct mask beats no mask ✓; no mask has higher ppl ✓
- lr 3e-4 ppl 31 ✗ (real **2,315**)
- given `<|im_start|>`, no mask writes "the user prompt" ≈ ✓ (it wrote the system prompt)

---

## Formulas / cheat sheet

- **SFT loss:** `L = −(1/|K|) Σ_{t∈K} log p(x_{t+1} | x_≤t)`, where K = positions whose **next** token is an assistant token.
- **Shift:** `logits[:, :-1]` vs `labels[:, 1:]`. Hugging Face passes labels **unshifted** and shifts inside the model.
- **`ignore_index = -100`:** PyTorch's default. `gather` can't index -100, so swap in 0 and then multiply by the mask. Those positions contribute 0 and get zero gradient.
- **Padding:** pad `input_ids` with any id, labels with -100, and set the attention_mask to 0.
- **ChatML:** `<|im_start|>{role}\n{content}<|im_end|>\n`. At inference, end the prompt with `<|im_start|>assistant\n` (the "generation prompt").
- **Hyperparameters:** lr ~1e-5 to 2e-5 (full), 1–3 epochs. LoRA tolerates ~1e-4 to 5e-4 (11/08 used 5e-4).
- **Pipeline:** pre-train (knowledge) → SFT (format and behaviour) → preference tuning, RLHF or DPO (which answer is *better*).

---

## My code (what I wrote and why)

```python
labels.extend([IGNORE] * len(header_ids))         # structure: read, not graded
if role == "assistant":
    if mask_mode == "drop_im_end":
        labels.extend(body_ids[:-1]); labels.append(IGNORE)
    else:
        labels.extend(body_ids)                   # answer + <|im_end|>: graded
else:
    labels.extend([IGNORE] * len(body_ids))
```

```python
safe_labels = torch.where(shift_labels == IGNORE, 0, shift_labels)   # gather needs valid indices
selected = torch.gather(log_probs, -1, safe_labels.unsqueeze(-1)).squeeze(-1)
valid = shift_labels != IGNORE
loss = (-selected * valid).sum() / valid.sum().clamp(min=1)          # mean over KEPT tokens
```

**Improvement still to make:** `body_ids[:-1]` assumes `<|im_end|>` is the last single id of `encode(content + IM_END)`. Encode `content` and `IM_END` **separately** so ownership is explicit; that is safe if a tokenizer adds EOS or splits the marker.

---

## Where this shows up in AI engineering

- **Project 3 (LoRA SFT + DPO):** this exact mask + template is what TRL's `SFTTrainer` (completion-only loss) does. Check the masking by decoding the kept labels of one batch before training.
- **Chat template mismatch** between training and serving (Llama-3 vs ChatML, a missing generation prompt) is a top cause of "the fine-tuned model rambles or never stops."
- **Interviews:** "Base vs instruct?", "Why mask the prompt?", "What is a chat template?", "How much data?", "SFT vs RAG?" (behaviour vs knowledge), "Why does SFT forget, and how do you prevent it?"
- **Eval:** check stop rate, held-out accuracy **and** a forgetting metric. Training loss alone hid the lr 3e-4 disaster.

---

## Mistakes to avoid

- **Masking the end token:** the model never learns to stop (0/45).
- **No masking:** it learns to write user and system turns, dilutes the answer signal, and forgets more.
- **Averaging over B·T instead of kept tokens:** short answers get under-weighted, and the loss scale depends on padding.
- **Labelling padding:** with pad = EOS, the model learns to spam EOS.
- **A pre-training lr (3e-4):** memorization plus catastrophic forgetting (ppl 2,315) while the training loss looks perfect.
- **Thinking SFT adds facts:** test answers like Damascus came from pre-training. Use RAG for new knowledge.
- **Mis-sizing gradient changes:** a behaviour that's never graded doesn't show up "half the time"; it shows up ~never.

---

## Words

- **Base model:** pre-trained only; continues text.
- **SFT / instruction tuning:** supervised fine-tuning on (instruction, response) conversations.
- **Chat template:** the exact text format, with special tokens, for roles and turns (ChatML, Llama-3, Alpaca).
- **Generation prompt:** the trailing `<|im_start|>assistant\n` that cues the model's turn at inference.
- **Loss masking / completion-only loss:** labels = -100 on non-response tokens.
- **Catastrophic forgetting:** losing general ability while fitting new data.
- **Special tokens:** new vocabulary rows (embedding resize) for template markers.

## Connects to

- [11-08 LoRA](11-08-lora.md): that experiment already used `labels = -100` on the prompt; today I built the mask myself. LoRA is the cheap way to run SFT.
- [07-07 GPT](07-07-gpt.md): same next-token loss and same shift; SFT only changes the data and the mask.
- [03-05 Loss Functions](03-05-loss-functions.md): cross-entropy = −log p(target). `ignore_index` just drops terms.
- [05-19 Subword Tokenization](05-19-subword-tokenization.md): special tokens are single ids, and segment-wise encoding keeps ownership clear.
- Next: 10/08 DPO, which teaches *which* answer is better (SFT can't).

---

## Self-test

1. **40-token prompt, 10-token answer, no mask: what share of the loss trains the answer?**
   <details><summary>Answer</summary>10/50 = 20%. The other 80% trains the model to write prompts.</details>
2. **Should the assistant's `<|im_end|>` be trained or masked? Why?**
   <details><summary>Answer</summary>Trained. It teaches the model to stop. When I masked it, the model stopped 0/45 times and looped.</details>
3. **Why divide by the number of kept tokens and not B·T?**
   <details><summary>Answer</summary>So the loss is a true per-response-token average, independent of prompt length and padding.</details>
4. **Why replace -100 with 0 before `gather`, and why is that safe?**
   <details><summary>Answer</summary>`gather` needs valid indices. The gathered value is multiplied by mask 0, so it adds nothing and gets zero gradient.</details>
5. **lr 3e-4 gave train loss 0.011 but 24/45 unseen and ppl 2,315. What happened?**
   <details><summary>Answer</summary>The big steps memorized 75 conversations and overwrote general language ability (catastrophic forgetting). Use a low lr and a held-out plus forgetting eval.</details>
6. **The SFT model answered "Damascus" though Syria was never in training. What does that show?**
   <details><summary>Answer</summary>SFT teaches format and behaviour; the knowledge came from pre-training.</details>
7. **Why can an SFT model have good perplexity yet worse chat quality?**
   <details><summary>Answer</summary>SFT imitates demonstrations token by token. It has no signal for which answer humans prefer, which needs RLHF or DPO.</details>
8. **How much SFT data is typical?**
   <details><summary>Answer</summary>10K–100K high-quality examples (Llama 2 Chat ~27K). Quality over quantity.</details>
