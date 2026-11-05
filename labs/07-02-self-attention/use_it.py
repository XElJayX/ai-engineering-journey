# Bridge lesson 12, "Use it": load YOUR weights into torch.nn.MultiheadAttention and make the numbers match.
# Run: python labs/07-02-self-attention/use_it.py   (needs torch)
import numpy as np
import torch
import torch.nn as nn

from my_attention import causal_mask, multi_head_attention

n, d_model, n_heads = 6, 8, 2
rng = np.random.default_rng(0)
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv, Wo = (rng.normal(size=(d_model, d_model)) / np.sqrt(d_model) for _ in range(4))

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, bias=False, batch_first=True).double()

# ---------------------------------------------------------------------------
# TODO 1: copy your weights into torch's layout.
# torch.nn.Linear-style layers compute  x @ W.T  (weight shape (out, in)), while yours compute  X @ W.
#   mha.in_proj_weight : shape (3*d_model, d_model) = Wq, Wk, Wv stacked on top of each other, in THEIR convention
#   mha.out_proj.weight: shape (d_model, d_model)   = Wo in THEIR convention
# Fill in the two numpy arrays below.
# ---------------------------------------------------------------------------
in_proj = np.vstack([Wq.T, Wk.T, Wv.T]) 
out_proj = Wo.T   

# ---------------------------------------------------------------------------
# TODO 2: torch's attn_mask uses the OPPOSITE boolean convention from yours:
#   yours: True = "allowed to look"        torch: True = "NOT allowed to look" (masked out)
# Build torch's causal mask from your causal_mask(n).
# ---------------------------------------------------------------------------
torch_mask = ~causal_mask(n)  # TODO: a numpy bool array of shape (n, n)


# ---------------------------------------------------------------------------
# Runner. Don't edit below.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if in_proj is None or out_proj is None or torch_mask is None:
        raise SystemExit("Fill in in_proj, out_proj and torch_mask first.")
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.from_numpy(np.asarray(in_proj)))
        mha.out_proj.weight.copy_(torch.from_numpy(np.asarray(out_proj)))
    Xt = torch.from_numpy(X)[None]                       # add a batch dimension: (1, n, d_model)
    for label, mine_mask, t_mask in [("no mask", None, None), ("causal", causal_mask(n), torch_mask)]:
        mine, my_heads = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=mine_mask)
        with torch.no_grad():
            out, w = mha(Xt, Xt, Xt, attn_mask=None if t_mask is None else torch.from_numpy(np.asarray(t_mask)),
                         need_weights=True, average_attn_weights=False)
        diff_out = np.abs(out[0].numpy() - mine).max()
        diff_w = max(np.abs(w[0, h].numpy() - my_heads[h]).max() for h in range(n_heads))
        verdict = "MATCH" if diff_out < 1e-10 and diff_w < 1e-10 else "MISMATCH"
        print(f"  {label:8}: max |torch - mine| output {diff_out:.1e}, per-head weights {diff_w:.1e}  -> {verdict}")
