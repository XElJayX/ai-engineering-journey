# Lesson 11/08 LoRA & QLoRA: a LoRA layer, injection into a model, merging, and 4-bit block quantization (int4 vs NF4).
# Run: python labs/11-08-lora/my_lora.py      (self-checks are small torch tensors; no model download)
# The "Use it" run (experiment.py) puts YOUR LoRA into GPT-2 small, teaches it a new output format, merges it,
# and quantizes GPT-2's real weights with YOUR int4 and NF4.
import math

import torch
import torch.nn as nn

# The 16 NF4 levels from the QLoRA paper (Dettmers et al. 2023): quantiles of a standard normal, rescaled to [-1, 1].
NF4_LEVELS = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                           -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                           0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                           0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


# ---------------------------------------------------------------------------
# Your TODOs
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        # TODO 1a: keep `base` (as self.base) and FREEZE its parameters (requires_grad = False).
        #   self.r, self.scaling = r, alpha / r
        #   self.A = nn.Parameter of shape (r, in_features),  init torch.randn(...) / math.sqrt(in_features)
        #   self.B = nn.Parameter of shape (out_features, r), init zeros
        #   (Same layout as nn.Linear.weight, which is (out, in): so B @ A has W's shape.)
        self.base = base
        device = self.base.weight.device
        dtype = self.base.weight.dtype
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r

        self.A = nn.Parameter(torch.randn(self.r, self.base.in_features, device=device, dtype=dtype)/ math.sqrt(self.base.in_features))
        self.B = nn.Parameter(torch.zeros(self.base.out_features, self.r, device=device, dtype=dtype))


    def forward(self, x):
        # TODO 1b: base(x) + scaling * (low-rank path). x is (..., in_features).
        #   Do NOT build the full (out, in) matrix B @ A here; go through the rank-r bottleneck.
        base_output = self.base(x)
        low_rank = (x@ self.A.T) @ self.B.T
        return base_output + self.scaling * low_rank

    def merged(self):
        # TODO 2: return a NEW plain nn.Linear (same in/out/bias) whose weight = W + scaling * (B @ A).
        #   Copy the bias. Work under torch.no_grad(). Don't modify self.base in place.
        new_layer = nn.Linear(self.base.in_features,self.base.out_features, bias = self.base.bias is not None, device = self.base.weight.device, dtype = self.base.weight.dtype)

        with torch.no_grad():
            merged_weight = self.base.weight + self.scaling * ( self.B @ self.A)
            new_layer.weight.copy_(merged_weight)
        
            if self.base.bias is not None:
                new_layer.bias.copy_(self.base.bias) 

        return new_layer


def inject_lora(model, targets, r=8, alpha=16):
    # TODO 3: freeze EVERY parameter of the model, then replace each nn.Linear whose LAST name component is in
    #   `targets` (e.g. "blocks.0.attn.q" -> "q") with LoRALinear(that_linear, r, alpha).
    #   Return the sorted list of full names you replaced.
    #   Hints: list(model.named_modules()) first (don't mutate while iterating); find the parent with
    #   model.get_submodule(parent_name) ("" means the model itself); swap with setattr(parent, child_name, new).
    for p in model.parameters():
        p.requires_grad = False

    replaced =[]

    for name,module in list(model.named_modules()):

        if not isinstance(module, nn.Linear):
            continue
        
        child_name = name.split(".")[-1]
        if child_name not in targets:
            continue
        parts = name.rsplit(".",1)

        parent_name = parts[0] if len(parts) == 2 else ""
        child_name = parts[-1]

        parent = model.get_submodule(parent_name)

        new_module = LoRALinear(module, r=r,alpha=alpha)
        setattr(parent,child_name,new_module)

        replaced.append(name)
    return sorted(replaced)




def count_params(model):
    # TODO 4: return (trainable, total) parameter counts as ints.

    trainable = 0
    total = 0

    for p in model.parameters():
        count = p.numel()
        if p.requires_grad:
            trainable+= count
        total+=count
    return (trainable,total)

def quantize_int4(w, block=64):
    # TODO 5: simulate absmax 4-bit quantization and return the DEQUANTIZED tensor (same shape as w).
    #   Flatten, split into blocks of `block` values (w.numel() is a multiple of block here).
    #   Per block: scale = max|w| / 7  (clamp to >= 1e-12);  q = round(w / scale).clamp(-8, 7);  out = q * scale.
    #   Evenly spaced levels: as many levels out at the edges as near zero.
    blocks = w.reshape(-1, block)

    max_abs = blocks.abs().amax(dim =1, keepdim = True)

    scale = (max_abs / 7).clamp_min(1e-12)

    q = (blocks/scale).round().clamp(-8,7)
    out = q * scale
    return out.reshape(w.shape)

def quantize_nf4(w, block=64):
    # TODO 6: same blocking, but NF4. Per block: scale = max|w| (clamp to >= 1e-12); x = w / scale lies in [-1, 1];
    #   replace each x by the NEAREST of the 16 NF4_LEVELS; out = level * scale. Return the dequantized tensor.
    #   Hint: distance matrix (n, 16) = (x[:, None] - NF4_LEVELS[None, :]).abs(), then argmin over the last dim.
    blocks = w.reshape(-1, block)
    max_abs = blocks.abs().amax(dim =1 , keepdim = True) 
    scale = max_abs.clamp_min(1e-12)
    normalized = blocks / scale
    levels = NF4_LEVELS.to(device=w.device, dtype=w.dtype)

    distances = (
    normalized.unsqueeze(-1) - levels
    ).abs()

    indices = distances.argmin(dim=-1)

    quantized = levels[indices]

    out = quantized * scale

    return out.reshape(w.shape)


# ---------------------------------------------------------------------------
# Self-checks. Don't edit below.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   ({detail})" if detail and not cond else ""))
    return cond


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.ModuleDict({"q": nn.Linear(d, d), "k": nn.Linear(d, d), "v": nn.Linear(d, d)})
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        a = self.attn["q"](x) * torch.sigmoid(self.attn["k"](x)) + self.attn["v"](x)
        return x + self.mlp(a)


class _Toy(nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.embed = nn.Linear(8, d)
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n)])
        self.head = nn.Linear(d, 3)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    ok = []
    base = nn.Linear(768, 2304)
    lin = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(4, 768)
    ok.append(check("1 shapes A (r,in) and B (out,r)", tuple(lin.A.shape) == (8, 768) and tuple(lin.B.shape) == (2304, 8)))
    ok.append(check("1 only A and B train: 8*(768+2304) = 24,576", sum(p.numel() for p in lin.parameters() if p.requires_grad) == 24576
                    and not any(p.requires_grad for p in lin.base.parameters())))
    ok.append(check("1 at init the layer IS the base layer", torch.equal(lin(x), base(x))))
    lin(x).pow(2).sum().backward()
    ok.append(check("1 first step: B gets a gradient, A gets exactly zero", lin.B.grad is not None and lin.B.grad.abs().sum() > 0
                    and lin.A.grad is not None and lin.A.grad.abs().sum() == 0))
    with torch.no_grad():
        lin.B.normal_(0, 0.02)
    want = x @ (base.weight + 2.0 * lin.B @ lin.A).T + base.bias
    ok.append(check("1 forward = x (W + (alpha/r) BA)^T + b", torch.allclose(lin(x), want, atol=1e-4), f"max diff {(lin(x) - want).abs().max():.2e}"))
    x3 = torch.randn(2, 5, 768)
    ok.append(check("1 works on (batch, seq, d) inputs", lin(x3).shape == (2, 5, 2304)))

    m = lin.merged()
    ok.append(check("2 merged is a plain nn.Linear", type(m) is nn.Linear and m.weight.shape == base.weight.shape))
    ok.append(check("2 merged == unmerged (to 1e-5)", torch.allclose(m(x), lin(x), atol=1e-5), f"max diff {(m(x) - lin(x)).abs().max():.2e}"))
    ok.append(check("2 base weight untouched", torch.allclose(lin.base.weight, base.weight) and not torch.allclose(m.weight, base.weight)))

    toy = _Toy()
    xt = torch.randn(5, 8)
    before = toy(xt).detach()
    total_before = sum(p.numel() for p in toy.parameters())
    names = inject_lora(toy, ["q", "v"], r=4, alpha=8)
    ok.append(check("3 replaces exactly the q and v linears", names == ["blocks.0.attn.q", "blocks.0.attn.v", "blocks.1.attn.q", "blocks.1.attn.v"], str(names)))
    ok.append(check("3 they are LoRALinear now", isinstance(toy.blocks[1].attn["v"], LoRALinear) and isinstance(toy.blocks[1].attn["k"], nn.Linear)))
    tr, tot = count_params(toy)
    ok.append(check("4 trainable = 4 layers * 4*(16+16) = 512", tr == 512, f"trainable {tr}"))
    ok.append(check("4 total = original + LoRA", tot == total_before + 512, f"{tot} vs {total_before + 512}"))
    ok.append(check("3 output unchanged right after injection", torch.allclose(toy(xt), before)))
    ok.append(check("3 embed/head/mlp/k stay frozen", not any(p.requires_grad for n_, p in toy.named_parameters() if ".A" not in n_ and ".B" not in n_)))

    w = torch.randn(256, 64)
    q4 = quantize_int4(w)
    ok.append(check("5 int4 keeps the shape", q4.shape == w.shape))
    ok.append(check("5 int4: <= 16 distinct values per block", max(len(torch.unique(b)) for b in q4.reshape(-1, 64)) <= 16))
    ok.append(check("5 int4 keeps each block's max exactly", torch.allclose(q4.reshape(-1, 64).abs().max(1).values, w.reshape(-1, 64).abs().max(1).values, atol=1e-6)))
    n4 = quantize_nf4(w)
    blk, s = n4.reshape(-1, 64), w.reshape(-1, 64).abs().max(1, keepdim=True).values
    ok.append(check("6 nf4 values are NF4 levels x block scale", torch.allclose((blk / s).unsqueeze(-1).sub(NF4_LEVELS).abs().min(-1).values, torch.zeros(1), atol=1e-6)))
    e4, en = (q4 - w).pow(2).mean().item(), (n4 - w).pow(2).mean().item()
    ok.append(check("6 on normal weights NF4 error < int4 error", en < e4, f"nf4 {en:.5f} vs int4 {e4:.5f}"))
    print(f"   (MSE on N(0,1) weights: int4 {e4:.5f}, nf4 {en:.5f})")

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
