# Use It for bridge lesson 14: your KV cache and tiled attention vs PyTorch and a real GPT-2.
# Run: python labs/07-12-kv-cache/use_it.py   (GPT-2 is already downloaded from lesson 13)
# transformers 5.x: model(..., use_cache=True) returns out.past_key_values, a DynamicCache.
#   cache.layers[i].keys / .values are tensors of shape (batch, n_kv_heads, seq_len, d_head).
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from my_kv import kv_cache_bytes, tiled_attention  # your code

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F


def torch_attention(q, K, V):
    # NumPy -> PyTorch, with batch/head/sequence dimensions
    q_t = torch.from_numpy(q).reshape(1, 1, 1, -1)
    K_t = torch.from_numpy(K).reshape(1, 1, K.shape[0], K.shape[1])
    V_t = torch.from_numpy(V).reshape(1, 1, V.shape[0], V.shape[1])

    out = F.scaled_dot_product_attention(
        q_t, K_t, V_t,
        is_causal=False
    )

    return out.reshape(-1).numpy()


@torch.no_grad()
def greedy_no_cache(ids, n_new):
    generated = []

    for _ in range(n_new):
        out = model(ids, use_cache=False)

        last_logits = out.logits[:, -1, :]
        next_id = torch.argmax(last_logits, dim=-1, keepdim=True)

        generated.append(next_id.item())
        ids = torch.cat([ids, next_id], dim=1)

    return generated


@torch.no_grad()
def greedy_cached(ids, n_new):
    generated = []

    # Prefill the prompt once
    out = model(ids, use_cache=True)
    cache = out.past_key_values

    for i in range(n_new):
        last_logits = out.logits[:, -1, :]
        next_id = torch.argmax(last_logits, dim=-1, keepdim=True)

        generated.append(next_id.item())

        # No need to process the final generated token
        if i < n_new - 1:
            out = model(
                next_id,
                past_key_values=cache,
                use_cache=True
            )
            cache = out.past_key_values

    return generated, cache


def actual_cache_bytes(cache):
    total = 0

    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()

    return total


# ---------------------------------------------------------------------------
# Harness. Don't edit below.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    q, K, V = rng.normal(size=8), rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
    diff = np.abs(torch_attention(q, K, V) - tiled_attention(q, K, V, tile=7)).max()
    print(f"A  your tiled attention vs torch SDPA: max abs diff {diff:.2e}  {'PASS' if diff < 1e-12 else 'FAIL'}")

    ids = tok("The KV cache is the reason that", return_tensors="pt").input_ids
    N = 150
    t0 = time.perf_counter(); slow = greedy_no_cache(ids, N); t_slow = time.perf_counter() - t0
    t0 = time.perf_counter(); fast, cache = greedy_cached(ids, N); t_fast = time.perf_counter() - t0
    print(f"B  same tokens with and without the cache: {'PASS' if slow == fast else 'FAIL'}")
    print(f"   no cache {t_slow:.2f}s   cached {t_fast:.2f}s   speedup {t_slow / t_fast:.1f}x")
    print("   text:", repr(tok.decode(fast[:25])), "...")

    seq = cache.get_seq_length()
    real = actual_cache_bytes(cache)
    pred = kv_cache_bytes(seq, n_layers=12, n_kv_heads=12, d_head=64, dtype_bytes=4)
    print(f"C  cache holds {seq} tokens: actual {real:,} bytes, your formula {pred:,}  {'PASS' if real == pred else 'FAIL'}")

    # Spot the silent bug: a teammate re-feeds the FULL ids together with the cache.
    with torch.no_grad():
        ids = tok("The KV cache stores", return_tensors="pt").input_ids
        out = model(ids, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        full = torch.cat([ids, nxt], dim=1)
        buggy = model(full, past_key_values=out.past_key_values, use_cache=True)
        right = model(full, use_cache=False)
    print("\nBUG  re-feeding full ids + cache:")
    print(f"     cache length now {buggy.past_key_values.get_seq_length()} (prompt + 1 = {full.shape[1]})")
    print(f"     next token: buggy {tok.decode(buggy.logits[0, -1].argmax())!r}   correct {tok.decode(right.logits[0, -1].argmax())!r}")
