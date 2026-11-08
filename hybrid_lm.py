"""Hybrid language model: shared depth, sparse experts, adapters, adaptive exits.

Cost levers:
  shared depth   - a few unique blocks run for several loops; params stay flat.
  MoE MLP        - N experts, top-k routing; total capacity grows, active MLP
                   compute stays at k experts per token.
  per-loop adapters - small zero-initialized bottlenecks give each loop its own
                   transformation cheaply.
  adaptive exits - each loop produces logits and is supervised. Inference can
                   stop early; KV cache makes cached decoding cheap.

Training data defaults to Python stdlib code.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

from language_model import CharTokenizer, RotaryEmbedding, TiktokenTokenizer


@dataclass
class Config:
    block_size: int = 256
    n_layer: int = 4
    n_loop: int = 2
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.1
    vocab_size: int = 0
    n_experts: int = 4
    top_k: int = 1
    moe_aux: float = 0.01
    router_z: float = 0.001
    adapter_dim: int = 16


class GQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.n_head % config.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.repeat = config.n_head // config.n_kv_head
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv_proj = nn.Linear(
            config.n_embd, 2 * self.n_kv_head * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, past_kv=None, use_cache=False):
        b, t, c = x.shape
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        if t > 1 and past_len > 0:
            raise ValueError("chunked prefill is not supported")

        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(b, t, 2, self.n_kv_head, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_cache = (k, v) if use_cache else None

        k_rep = k.repeat_interleave(self.repeat, dim=1)
        v_rep = v.repeat_interleave(self.repeat, dim=1)
        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(t > 1 and past_len == 0),
        )
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.o_proj(out), new_cache


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.fc1 = nn.Linear(config.n_embd, config.adapter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.adapter_dim, config.n_embd)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.ln(x))))


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.mlp_ratio * config.n_embd)
        self.fc2 = nn.Linear(config.mlp_ratio * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = min(config.top_k, config.n_experts)
        self.router = nn.Linear(config.n_embd, self.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.n_experts)])
        self.aux = 0.0
        self.z_loss = 0.0

    def forward(self, x):
        b, t, c = x.shape
        flat = x.reshape(-1, c)
        logits = self.router(flat)
        probs = logits.softmax(dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            for slot in range(self.top_k):
                selected = indices[:, slot] == expert_id
                token_idx = selected.nonzero(as_tuple=False).squeeze(1)
                if token_idx.numel() == 0:
                    continue
                expert_out = expert(flat[token_idx])
                w = weights[token_idx, slot].unsqueeze(-1)
                out.index_add_(0, token_idx, expert_out * w)

        assignment = F.one_hot(indices[:, 0], self.n_experts).float()
        frac = assignment.mean(dim=0)
        mean_prob = probs.mean(dim=0)
        self.aux = self.n_experts * (frac * mean_prob).sum()
        self.z_loss = logits.logsumexp(dim=-1).pow(2).mean()
        return out.view(b, t, c)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = GQAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.moe = MoEMLP(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.moe(self.ln2(x)))
        return x, new_cache


class HybridGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.loop_emb = nn.Parameter(torch.zeros(config.n_loop, config.n_embd))
        self.adapters = None
        if config.adapter_dim > 0:
            self.adapters = nn.ModuleList(
                [
                    nn.ModuleList([Adapter(config) for _ in range(config.n_layer)])
                    for _ in range(config.n_loop)
                ]
            )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        if self.adapters is not None:
            for loop_adapters in self.adapters:
                for adapter in loop_adapters:
                    nn.init.zeros_(adapter.fc2.weight)
                    nn.init.zeros_(adapter.fc2.bias)
        effective_depth = config.n_layer * config.n_loop
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * effective_depth)
                )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def param_metrics(self):
        """Return (unique active params, parameter applications per token)."""
        emb = self.tok_emb.weight.numel()
        block = self.blocks[0]
        per_block = sum(p.numel() for p in block.attn.parameters())
        per_block += sum(p.numel() for p in block.moe.experts[0].parameters())
        per_block += sum(p.numel() for p in block.moe.router.parameters())
        per_block += sum(p.numel() for p in block.ln1.parameters())
        per_block += sum(p.numel() for p in block.ln2.parameters())
        if self.adapters is not None:
            adapters = sum(p.numel() for p in self.adapters.parameters())
        else:
            adapters = 0
        common = emb + self.loop_emb.numel() + sum(
            p.numel() for p in self.ln_f.parameters()
        )
        unique_active = (
            common + self.config.n_layer * per_block + adapters
        )
        applied = (
            common
            + self.config.n_loop * self.config.n_layer * per_block
            + adapters
        )
        return unique_active, applied

    def forward(self, idx, targets=None, exit_threshold=None, max_loops=None,
                caches=None, use_cache=False):
        config = self.config
        x = self.drop(self.tok_emb(idx))
        n_loops = config.n_loop if max_loops is None else min(max_loops, config.n_loop)
        if use_cache and caches is None:
            caches = [None] * (config.n_loop * config.n_layer)

        loop_losses, logits = [], None
        aux_sum, z_sum, active_loops, exited = 0.0, 0.0, 0, False

        for loop in range(n_loops):
            x = x + self.loop_emb[loop]
            for i, block in enumerate(self.blocks):
                cache_idx = loop * config.n_layer + i
                past_kv = caches[cache_idx] if use_cache else None
                x, new_cache = block(x, past_kv=past_kv, use_cache=use_cache)
                if use_cache:
                    caches[cache_idx] = new_cache
                if self.adapters is not None:
                    x = x + self.adapters[loop][i](x)
                aux_sum = aux_sum + block.moe.aux
                z_sum = z_sum + block.moe.z_loss

            logits = self.head(self.ln_f(x))
            active_loops = loop + 1
            if targets is not None:
                loop_losses.append(
                    F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                    )
                )
            if exit_threshold is not None and loop < n_loops - 1:
                confidence = logits.detach().softmax(dim=-1).max(dim=-1).values.mean()
                if confidence.item() >= exit_threshold:
                    exited = True
                    break

        loss = None
        if targets is not None:
            loss = torch.stack(loop_losses).mean()
            aux = aux_sum / max(active_loops * config.n_layer, 1)
            z = z_sum / max(active_loops * config.n_layer, 1)
            loss = loss + config.moe_aux * aux + config.router_z * z

        info = {"loops": active_loops, "exited": exited}
        if use_cache:
            info["caches"] = caches
        return logits, loss, info

    @staticmethod
    def _sample(logits, temperature, top_k):
        if temperature <= 0:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
        return token

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50,
                 exit_threshold=None, use_cache=True, depth=None):
        self.eval()
        config = self.config
        prompt_len = idx.size(1)
        room = config.block_size - prompt_len
        if room <= 0:
            raise ValueError("prompt is longer than the context window")
        max_new_tokens = min(max_new_tokens, room)
        loops_used = []
        if max_new_tokens == 0:
            return idx, loops_used

        if use_cache:
            if depth is not None:
                logits, _, info = self(idx, max_loops=depth, use_cache=True)
            elif exit_threshold is not None:
                logits, _, info = self(idx, exit_threshold=exit_threshold, use_cache=True)
            else:
                logits, _, info = self(idx, use_cache=True)
            n_loops = info["loops"]
            caches = info["caches"][: n_loops * config.n_layer]
            next_logits = logits[:, -1, :]

            for step in range(max_new_tokens):
                loops_used.append(n_loops)
                next_token = self._sample(next_logits, temperature, top_k)
                idx = torch.cat([idx, next_token], dim=1)
                if step == max_new_tokens - 1:
                    break
                x = self.drop(self.tok_emb(next_token))
                for loop in range(n_loops):
                    x = x + self.loop_emb[loop]
                    for i, block in enumerate(self.blocks):
                        cache_idx = loop * config.n_layer + i
                        x, new_cache = block(
                            x, past_kv=caches[cache_idx], use_cache=True
                        )
                        caches[cache_idx] = new_cache
                        if self.adapters is not None:
                            x = x + self.adapters[loop][i](x)
                    next_logits = self.head(self.ln_f(x))[:, -1, :]
        else:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -config.block_size:]
                logits, _, info = self(
                    idx_cond, exit_threshold=exit_threshold, max_loops=depth
                )
                loops_used.append(info["loops"])
                next_logits = logits[:, -1, :]
                next_token = self._sample(next_logits, temperature, top_k)
                idx = torch.cat([idx, next_token], dim=1)
        return idx, loops_used


def load_tokenizer(name, texts):
    if name == "char":
        return CharTokenizer("\n".join(texts))
    return TiktokenTokenizer("gpt2")


def get_batch(data, block_size, batch_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def loop_logits(model, x):
    hidden = model.drop(model.tok_emb(x))
    out = []
    for loop in range(model.config.n_loop):
        hidden = hidden + model.loop_emb[loop]
        for i, block in enumerate(model.blocks):
            hidden, _ = block(hidden)
            if model.adapters is not None:
                hidden = hidden + model.adapters[loop][i](hidden)
        out.append(model.head(model.ln_f(hidden)))
    return out


@torch.no_grad()
def evaluate(model, data, config, batch_size, device, iters=20):
    model.eval()
    losses = torch.zeros(iters)
    loop_losses = [[] for _ in range(config.n_loop)]
    for i in range(iters):
        x, y = get_batch(data, config.block_size, batch_size, device)
        logits_list = loop_logits(model, x)
        for loop, logits in enumerate(logits_list):
            loop_losses[loop].append(
                F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            )
        losses[i] = loop_losses[-1][-1]
    model.train()
    return losses.mean().item(), [sum(v) / len(v) for v in loop_losses]


@torch.no_grad()
def calibrate_exits(model, data, config, batch_size, device, iters=10,
                    thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    model.eval()
    totals = {
        t: {"loss": 0.0, "count": 0, "loops": 0.0, "early": 0.0}
        for t in thresholds
    }
    loop_totals = [0.0] * config.n_loop
    count = 0
    for _ in range(iters):
        x, y = get_batch(data, config.block_size, batch_size, device)
        logits_list = loop_logits(model, x)
        batch = x.size(0)
        count += batch
        per_sample_ce = []
        confidences = []
        for logits in logits_list:
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).view(batch, -1).mean(dim=1)
            per_sample_ce.append(ce)
            confidences.append(logits.softmax(dim=-1).max(dim=-1).values.mean(dim=1))
        ce_all = torch.stack(per_sample_ce)
        conf_all = torch.stack(confidences)

        for t in thresholds:
            meets = conf_all >= t
            any_meet = meets.any(dim=0)
            chosen = torch.where(
                any_meet, meets.float().argmax(dim=0),
                torch.full_like(meets[0], config.n_loop - 1, dtype=torch.long),
            )
            selected = ce_all.gather(0, chosen.unsqueeze(0)).squeeze(0)
            totals[t]["loss"] += selected.sum().item()
            totals[t]["count"] += batch
            totals[t]["loops"] += (chosen + 1).float().sum().item()
            totals[t]["early"] += (chosen < config.n_loop - 1).sum().item()
        for loop in range(config.n_loop):
            loop_totals[loop] += per_sample_ce[loop].mean().item() * batch
    model.train()

    curve = []
    for t in thresholds:
        r = totals[t]
        curve.append(
            {
                "threshold": t,
                "val_loss": r["loss"] / max(r["count"], 1),
                "avg_loops": r["loops"] / max(r["count"], 1),
                "early_exit_fraction": r["early"] / max(r["count"], 1),
            }
        )
    return curve, [v / max(count, 1) for v in loop_totals]


def lr_at(step, total, lr, warmup, min_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def run_experiment(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    train_text = Path(args.train_data).read_text(encoding="utf-8")
    val_text = Path(args.val_data).read_text(encoding="utf-8")
    tokenizer = load_tokenizer(args.tokenizer, [train_text, val_text])
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    config = Config(
        block_size=args.block_size,
        n_layer=args.layers,
        n_loop=args.loops,
        n_head=args.heads,
        n_kv_head=args.kv_heads,
        n_embd=args.embd,
        dropout=args.dropout,
        vocab_size=tokenizer.vocab_size,
        n_experts=args.experts,
        top_k=args.top_k,
        moe_aux=args.moe_aux,
        router_z=args.router_z,
        adapter_dim=args.adapter_dim,
    )
    model = HybridGPT(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    unique_active, applied = model.param_metrics()
    print(
        f"[{args.name}] tokens train/val {len(train_data):,}/{len(val_data):,} | "
        f"vocab {config.vocab_size:,}\n"
        f"[{args.name}] {config.n_layer} layers x {config.n_loop} loops = "
        f"{config.n_layer * config.n_loop} effective | experts {config.n_experts} top-{config.top_k} | "
        f"adapters {config.adapter_dim} | GQA {config.n_head}q/{config.n_kv_head}kv\n"
        f"[{args.name}] params total {total:,} | unique active {unique_active:,} | "
        f"applications/token {applied:,} | device {device}",
        flush=True,
    )

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95),
    )

    model.train()
    start = time.time()
    tokens_seen = 0
    last_loss = None
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, config.block_size, args.batch_size, device)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += x.numel()
        last_loss = loss.item()

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start
            print(
                f"[{args.name}] step {step + 1:5d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {tokens_seen / elapsed:,.0f} tok/s",
                flush=True,
            )
        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            val_loss, by_loop = evaluate(
                model, val_data, config, args.batch_size, device, args.eval_iters
            )
            loops = " ".join(f"L{i + 1} {v:.4f}" for i, v in enumerate(by_loop))
            print(
                f"[{args.name}] step {step + 1:5d} | val {val_loss:.4f} | {loops}",
                flush=True,
            )

    train_seconds = time.time() - start
    val_loss, by_loop = evaluate(
        model, val_data, config, args.batch_size, device, args.eval_iters
    )
    exit_curve, loop_ce = calibrate_exits(
        model, val_data, config, args.batch_size, device, args.calibrate_iters
    )
    print(f"[{args.name}] exit calibration:", flush=True)
    for row in exit_curve:
        print(
            f"  threshold {row['threshold']:.2f} | val {row['val_loss']:.4f} | "
            f"loops {row['avg_loops']:.2f} | early {row['early_exit_fraction']:.1%}",
            flush=True,
        )

    # Compare exits against the full-depth loss on the same calibration batches;
    # by_loop comes from a different random sample and adds noise.
    full_loss = loop_ce[-1]
    tolerance = max(0.005, 0.01 * full_loss)
    candidates = [
        row for row in exit_curve
        if row["avg_loops"] < config.n_loop - 0.01
        and row["val_loss"] <= full_loss + tolerance
    ]
    if candidates:
        selected_exit = min(candidates, key=lambda row: row["avg_loops"])
        sample_threshold = selected_exit["threshold"]
    else:
        selected_exit = {
            "threshold": None,
            "val_loss": full_loss,
            "avg_loops": float(config.n_loop),
            "early_exit_fraction": 0.0,
        }
        sample_threshold = None
    print(
        f"[{args.name}] selected exit: threshold {selected_exit['threshold']} | "
        f"val {selected_exit['val_loss']:.4f} | loops {selected_exit['avg_loops']:.2f}",
        flush=True,
    )

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    sample_out, sample_loops = model.generate(
        prompt_ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k_sample,
        exit_threshold=sample_threshold, use_cache=True,
    )
    sample_text = tokenizer.decode(sample_out[0].tolist())

    bench_tokens = min(64, config.block_size - prompt_ids.size(1))
    t0 = time.time()
    model.generate(
        prompt_ids, max_new_tokens=bench_tokens, temperature=0.8, top_k=50,
        depth=config.n_loop, use_cache=True,
    )
    cached_tok_s = bench_tokens / max(time.time() - t0, 1e-6)
    t0 = time.time()
    model.generate(
        prompt_ids, max_new_tokens=bench_tokens, temperature=0.8, top_k=50,
        depth=config.n_loop, use_cache=False,
    )
    plain_tok_s = bench_tokens / max(time.time() - t0, 1e-6)

    metrics = {
        "name": args.name,
        "config": asdict(config),
        "params_total": total,
        "params_active_unique": unique_active,
        "params_applied_per_token": applied,
        "steps": args.steps,
        "tokens_seen": tokens_seen,
        "train_seconds": train_seconds,
        "tokens_per_sec": tokens_seen / max(train_seconds, 1e-6),
        "train_loss_final": last_loss,
        "val_loss": val_loss,
        "loop_val_losses": by_loop,
        "exit_curve": exit_curve,
        "exit_loop_ce": loop_ce,
        "selected_exit": selected_exit,
        "sample": sample_text,
        "sample_loops_mean": sum(sample_loops) / max(len(sample_loops), 1),
        "sample_exit_threshold": sample_threshold,
        "gen_cached_tok_s": cached_tok_s,
        "gen_plain_tok_s": plain_tok_s,
        "device": device,
    }
    print(
        f"[{args.name}] final val {val_loss:.4f} | loops {' '.join(f'{v:.4f}' for v in by_loop)} | "
        f"gen cached {cached_tok_s:.1f} tok/s vs nocache {plain_tok_s:.1f} tok/s\n"
        f"--- sample | loops {metrics['sample_loops_mean']:.2f} ---\n{sample_text}\n--- end ---",
        flush=True,
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "tokenizer": args.tokenizer,
                "chars": getattr(tokenizer, "chars", None),
                "args": vars(args),
            },
            args.out,
        )
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hybrid")
    parser.add_argument("--train-data", default="data/code_train.txt")
    parser.add_argument("--val-data", default="data/code_val.txt")
    parser.add_argument("--tokenizer", choices=["char", "gpt2"], default="char")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--embd", type=int, default=128)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--adapter-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--moe-aux", type=float, default=0.01)
    parser.add_argument("--router-z", type=float, default=0.001)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--calibrate-iters", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--prompt", default="def ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k-sample", type=int, default=50)
    parser.add_argument("--exit-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="hybrid_code.pt")
    parser.add_argument("--metrics-out", default="hybrid_code_metrics.json")
    return parser.parse_args(argv)


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
