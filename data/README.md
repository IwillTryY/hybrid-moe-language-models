# Data

Raw corpora and trained checkpoints are **not** committed. The repository ships
the scripts needed to rebuild each dataset.

## Code corpus (main experiment)

```bash
python scripts/build_code_corpus.py
```

Reads Python files from the active Python installation's standard library,
excluding third-party `site-packages` and test directories. The reported run used
`data/code_split.json`, a manifest of the exact non-empty files assigned to each
split. The script reproduces that split byte-for-byte on the same Python installation.
If the manifest is removed, the script falls back to a seeded 90/10 file split.

Measured sizes with Python 3.14 on the development machine, manifest mode:

- train: 642 non-empty files, 11,098,501 characters
- validation: 71 non-empty files, 1,022,905 characters

The manifest is public file metadata only; no source code is redistributed.
Source license: Python Software Foundation License. This repository does not
redistribute the Python source; the local installation is the source of truth:

- https://docs.python.org/3/license.html
- https://www.python.org/psf/license/

Standard library versions differ across Python releases. Exact character counts
and token counts will vary slightly.

## WikiText-2 (optional natural-language runs)

```bash
pip install datasets
python scripts/download_wikitext.py
```

Fetches `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, from Hugging
Face, and writes `data/wikitext2_train.txt` and `data/wikitext2_val.txt`. Consult
the dataset card for license and citation requirements:

- https://huggingface.co/datasets/Salesforce/wikitext

## Tiny Shakespeare (baseline natural-language demo)

`language_model.py` downloads this corpus at runtime if `data/tinyshakespeare.txt`
is absent. The source file is public domain and hosted by the char-rnn repository:

- https://github.com/karpathy/char-rnn

## Privacy

No personal data, credentials, proprietary source, or private corpora are used.
Both corpora are public/open source. Model outputs in `results/` are synthetic text.
