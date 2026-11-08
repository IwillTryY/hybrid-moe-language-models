"""Build the Python source corpus used in the main experiment.

By default this uses data/code_split.json, which records the exact non-empty files
assigned to train and validation by the reported run. That makes the split
reproducible on the same Python installation. Without a manifest, files are
sorted, shuffled with seed 1337, and split 90/10.

Python source is distributed under the PSF License. No source files are committed;
this script reads a local installation and writes derived text.
"""

import argparse
import json
import random
import sysconfig
from pathlib import Path


def collect_files(root, include_tests):
    files = [
        p for p in root.rglob("*.py")
        if p.is_file() and "site-packages" not in p.parts
    ]
    if not include_tests:
        files = [p for p in files if "test" not in p.relative_to(root).parts]
    return sorted(files)


def write_corpus(root, out_dir, name, paths, missing):
    chunks = []
    for path in paths:
        full = root / path
        if not full.is_file():
            missing.append(path)
            continue
        text = full.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            chunks.append(f"\n# ==== {path} ====\n{text}")
    data = "".join(chunks)
    (out_dir / name).write_text(data, encoding="utf-8")
    print(f"{name}: {len(chunks):,} files, {len(data):,} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Python stdlib directory")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--manifest", default="data/code_split.json",
        help="exact split manifest; pass an empty string to regenerate a split",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include-tests", action="store_true")
    args = parser.parse_args()

    root = Path(args.root or sysconfig.get_paths()["stdlib"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest) if args.manifest else None
    missing = []
    if manifest is not None and manifest.is_file():
        split = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"using split manifest {manifest}")
        write_corpus(root, out_dir, "code_train.txt", split["train"], missing)
        write_corpus(root, out_dir, "code_val.txt", split["val"], missing)
    else:
        files = collect_files(root, args.include_tests)
        if not files:
            raise SystemExit(f"no Python files found under {root}")
        random.Random(args.seed).shuffle(files)
        n_val = max(1, int(args.val_fraction * len(files)))
        val_files, train_files = files[:n_val], files[n_val:]
        for name, group in (
            ("code_train.txt", train_files),
            ("code_val.txt", val_files),
        ):
            write_corpus(
                root, out_dir, name,
                [p.relative_to(root).as_posix() for p in group], missing,
            )

    if missing:
        print(f"warning: {len(missing)} manifest paths were not found in {root}")
        print("first missing:", missing[:5])


if __name__ == "__main__":
    main()
