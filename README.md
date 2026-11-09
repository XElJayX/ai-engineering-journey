# AI Engineering Journey

I'm training to be an AI engineer, working on LLM, RAG and agent systems. In this repo I build each piece from the math up, then **run it against the real library or a real model and measure what happens**. Every lab ends with an experiment and a number, and the numbers often surprised me.

**Jayanth Ravimurugan**, MS Data Science (graduating May 2027) · [Portfolio](https://jayanthravimurugan.vercel.app) · [GitHub](https://github.com/XElJayX)

## What I measured

All runs are local: llama3.2 (3B) on Ollama, BGE-small embeddings, a MiniLM cross-encoder, and GPT-2 (124M) fine-tuned on an Apple-silicon GPU. The corpora are fictional SEC-style filings, so the model can't answer from memory.

| Lab | Experiment | Result | What I took away |
|-----|-----------|--------|------------------|
| [10/08 DPO](labs/10-08-dpo/) | My own DPO and SimPO losses on GPT-2, teaching "concise > chatty" 7 ways from one SFT reference | DPO at lr 5e-7: concise 13% → 21–30%, SimPO 53%, SFT on the winners **100%**. At lr 5e-6 the loss hit **0.000** while the winner's own log-prob *fell* (−2.4) and the loser's dropped −100 nats, and the model collapsed: 0% concise, 46% correct. Adding an SFT term fixed it (99%). | Preference accuracy was 100% on every run, *including the collapsed one*, and perplexity barely moved. Read sampled generations, log the chosen log-prob, and run the SFT-on-winners baseline first. |
| [10/06 Instruction tuning (SFT)](labs/10-06-instruction-tuning-sft/) | Base GPT-2 → chat model with my own chat template, loss mask and masked loss; 4 variants, 45 unseen prompts | Correct mask: stops **45/45**, correct 37/45 (base: 0/45 and 5/45). Masking `<|im_end|>`: **0/45** stops. No mask: 33/45, and it learned to write system prompts. lr 3e-4: training loss 0.011 but English perplexity **26 → 2,315** | SFT teaches format and when to stop; the facts came from pre-training. Train the end token. A near-zero training loss can hide a destroyed model, so always run a held-out and a forgetting eval. |
| [11/08 LoRA & QLoRA](labs/11-08-lora/) | My own LoRA layer, injection, merge and int4/NF4 quantizers on GPT-2 | Raw injection silently replaced **0 layers** (Conv1D, not Linear). After the fix: 0.24% of params trainable, format **40/40** like full fine-tuning, Adam state 2.2 MB vs 949 MB. LoRA still raised perplexity 26.4 → 33.6. NF4 +1.3 ppl vs int4 +3.0 | Assert trainable params > 0 after injecting. A frozen W doesn't mean no forgetting, so measure it. NF4's normal-shaped levels beat evenly spaced 4-bit. |
| [11/07 Advanced RAG](labs/11-07-advanced-rag/) | Vector vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder rerank, 48 queries over 192 10-K chunks | Vector **39/48**, BM25 32, hybrid RRF **32** (worse than vector), hybrid + rerank **44/48** at 58 ms per query | RRF rewards agreement, so BM25 noise on paraphrased queries outvoted vector's correct #1. Use hybrid to retrieve wide, then rerank narrow. |
| [11/06 RAG](labs/11-06-rag/) | Plain chunks vs chunks with a `[Company 10-K FY]` header, 32 filings | **10/20 → 20/20** correct. Plain chunks gave 10 wrong dollar figures *with citations*. With no context at all, the model abstained 24/24 times. | Wrong context makes a model confidently wrong. Classify every failure as a retrieval failure or a generation failure. |
| [11/05 Context engineering](labs/11-05-context-engineering/) | 4,645-token prompt sent to Ollama with default settings | The server saw only **2,050 tokens**: it cut from the *front* silently, so the system rule and the best document were gone | Check that the tokens the server saw match the tokens you sent (≥ 0.95). Sending the top 24 docs instead of the top 3 cost 6.9× the latency for zero accuracy gain. |
| [11/04 Vector search](labs/11-04-embeddings/) | Exact vs binary-quantized vs IVF over 5,800 chunks | Binary recall 0.57, then **0.93** after rescoring the top 100. IVF recall 0.53 / 0.78 / 0.94 at nprobe 1 / 4 / 16 | Approximate search loses results with no error, so measure recall against exact search. Below ~100k vectors, exact search was faster (0.07 ms). |
| [11/03 Structured outputs](labs/11-03-structured-outputs/) | Prompt-only vs JSON mode vs JSON Schema, plus retry, on 6 filings | Prompt-only 0/6 valid, schema mode 6/6 valid but 5/6 correct. Retrying with the raw parser error: **0/6 in 18 calls**. With plain-English feedback: 6/6. | A schema guarantees the shape, not the truth (an injected value passed every mode). A required non-null field made the decoder *invent* `fiscal_year: 0`. |
| [11/02 Few-shot and CoT](labs/11-02-few-shot-cot/) | 4 prompting modes + self-consistency on 12 word problems | Zero-shot **0/12**, chain of thought **10/12**, self-consistency (n=5) 11/12 at 5× the tokens | Self-consistency confidence is a routing signal. A systematic arithmetic error survives voting, so hand it to a calculator tool. |
| [11/01 Prompt engineering](labs/11-01-prompt-engineering/) | Vague vs engineered prompt, with and without JSON mode | Valid JSON 0% → 67% → **100%**. Vague prompt + JSON mode: 100% valid, **0% correct keys** | JSON mode fixes syntax, and the prompt or schema fixes content. |
| [07/12 KV cache](labs/07-12-kv-cache/) | My own prefill/decode cache and tiled online-softmax attention vs PyTorch on GPT-2 | Matches SDPA to 1e-16. Same tokens **4.4× faster** with the cache | Llama-3-8B needs 128 KiB of cache per token, about 60 users per H100 at 8K context. Memory, not FLOPs, sets serving capacity. |
| [07/07 GPT](labs/07-07-gpt/) | My own GPT forward pass, loss and samplers vs Hugging Face GPT-2 | Loss 4.6375 matches. Greedy output is token-identical to `model.generate` | On a real model, top-p 0.9 kept **1,503** tokens and min-p 0.1 kept 16. |
| [07/02 Self-attention](labs/07-02-self-attention/) | My own multi-head attention loaded into `nn.MultiheadAttention` | Matches to 1e-15 | PyTorch stores weights transposed, and in its masks `True` means *blocked*. |
| [05/19 Tokenization](labs/05-19-subword-tokenization/) | My own BPE vs tiktoken o200k | A one-word Japanese input was **15 tokens**, and the family emoji was 11 | Count tokens, never estimate them from word counts. |
| [05/22 Embeddings](labs/05-22-embedding-models/), [05/23 Chunking](labs/05-23-chunking/) | Query prefix, truncation, hybrid search and chunk strategies with BGE-small | The query prefix added **+10 points** of recall@1. Recursive 200-token chunks gave hit@3 1.0 at 84 prompt tokens | Follow the model card. Score retrieval hits *and* prompt cost together. |

Earlier labs (in [`labs/`](labs/)): autograd and backprop, loss functions, PCA, sampling and temperature, information theory, eval metrics, imbalanced data.

## How each lab works

1. **Build it.** I write the algorithm myself in NumPy (`my_*.py`), and it passes a set of self-checks.
2. **Use it.** I run my version next to the production library or a live model, predict the result before running, and record what actually happened (`use_it.py` / `experiment.py`).
3. **Write it down.** One revision sheet per lab in [`notes/`](notes/), with worked numbers, the mistakes I made and a self-test.

```bash
pip install -r requirements.txt
python labs/11-07-advanced-rag/experiment.py   # ~1 min, no LLM needed
```

The LLM experiments need [Ollama](https://ollama.com) with `llama3.2` pulled. Lab 11/04 embeds the curriculum's own lesson docs: set `CURRICULUM_DIR` to a clone of the curriculum repo.

## Where it's heading

| Next | Target date |
|------|-------------|
| **Project 1:** RAG over SEC 10-K/10-Q filings. Hybrid retrieval + rerank, verified citations, a 150+ question golden set, and a CI gate that fails a pull request if eval scores regress. Deployed. | Dec 2026 |
| **Project 2:** SRE incident agent on my own MCP servers. Human approval, traces, and root-cause accuracy on 30+ seeded incidents. | Mar 2027 |
| **Project 3:** LoRA SFT + DPO on a small open model, quantized and served with vLLM. Quality vs cost vs latency compared with a frontier model. | May 2027 |

## Credits

The lab structure follows the open-source [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) curriculum (MIT). The implementations, experiments, results and notes in this repo are my own.
