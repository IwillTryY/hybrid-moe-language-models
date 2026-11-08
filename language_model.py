"""Small GPT-style language model: RoPE, causal attention, next-token training.

Default is character-level Tiny Shakespeare. Pass --tokenizer gpt2 to use the
GPT-2 BPE tokenizer from tiktoken instead.

    python language_model.py --steps 2000
    python language_model.py --tokenizer gpt2 --steps 2000 --prompt "ROMEO: "
"""

import argparse
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


@dataclass
class Config:
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    vocab_size: int = 0


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} is not in the training corpus") from None

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class TiktokenTokenizer:
    def __init__(self, name="gpt2"):
        import tiktoken

        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode(ids)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, offset=0):
        _, _, t, _ = x.shape
        freqs = torch.outer(
            torch.arange(offset, offset + t, device=x.device, dtype=self.inv_freq.dtype),
            self.inv_freq,
        )
        cos = freqs.cos()[None, None]
        sin = freqs.sin()[None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(
            2, 0, 3, 1, 4
        ).unbind(0)
        q = self.rope(q)
        k = self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def load_text(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


def get_batch(data, block_size, batch_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, eval_iters):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, total_steps, lr, warmup, min_lr_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tinyshakespeare.txt")
    parser.add_argument("--tokenizer", choices=["char", "gpt2"], default="char")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-frac", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--prompt", default="ROMEO: ")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="language_model.pt")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    text = load_text(args.data)
    if args.tokenizer == "char":
        tokenizer = CharTokenizer(text)
    else:
        tokenizer = TiktokenTokenizer("gpt2")

    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"text: {len(text):,} chars | tokens: {len(data):,} | vocab: {tokenizer.vocab_size:,}")

    config = Config(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        vocab_size=tokenizer.vocab_size,
    )
    model = GPT(config).to(device)
    print(
        f"model: {model.num_params():,} params "
        f"({model.num_params(non_embedding=True):,} non-embedding) | device: {device}"
    )

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    model.train()
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup, args.min_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(splits["train"], config.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_interval == 0 or step == 0:
            print(f"step {step + 1:5d} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)
        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            losses = estimate_loss(
                model, splits, config.block_size, args.batch_size, device, args.eval_iters
            )
            print(
                f"step {step + 1:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}",
                flush=True,
            )

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    sample = tokenizer.decode(out[0].tolist())
    print("\n--- sample ---")
    print(sample)
    print("--- end ---\n")

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
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
