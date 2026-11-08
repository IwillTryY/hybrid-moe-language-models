"""Load language_model.pt and generate text from a prompt."""

import argparse

import torch

from language_model import CharTokenizer, Config, GPT, TiktokenTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="language_model.pt")
    parser.add_argument("--prompt", default="ROMEO: ")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["tokenizer"] == "char":
        tokenizer = CharTokenizer("".join(checkpoint["chars"]))
    else:
        tokenizer = TiktokenTokenizer("gpt2")

    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
