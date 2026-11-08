"""Download WikiText-2 for optional natural-language runs.

Requires the optional `datasets` dependency. The dataset is not redistributed
with this repository; consult the dataset card for its license and citation.
"""

from pathlib import Path


def main():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "install the optional dependency: pip install datasets"
        ) from exc

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    for split, filename in (
        ("train", "wikitext2_train.txt"),
        ("validation", "wikitext2_val.txt"),
    ):
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
        )
        text = "\n".join(ds["text"])
        (out_dir / filename).write_text(text, encoding="utf-8")
        print(f"{filename}: {len(text):,} chars")


if __name__ == "__main__":
    main()
