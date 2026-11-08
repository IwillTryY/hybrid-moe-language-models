# Results and methodology

All numbers are from the code in this repository. The main result is a single-seed
pilot on character-level Python code. Treat it as evidence for a direction, not a
final benchmark.

## Main experiment: code language modeling

### Data

Python standard-library source, excluding `site-packages` and test directories.
The reported run used 642 non-empty training files (11.1M characters) and 71
non-empty validation files (1.02M characters), recorded in
`data/code_split.json`. Vocabulary is 277 characters. Separators at file boundaries
look like:

```python
# ==== asyncio/base_events.py ====
```

The exact split is reproduced by `scripts/build_code_corpus.py`, which reads the
committed manifest `data/code_split.json`. Without the manifest, the script falls
back to a seeded 90/10 file split.

### Training

| Setting | Value |
|---|---|
| Context | 256 characters |
| Batch size | 32 |
| Steps | 1,500 |
| Tokens seen | 12.3M |
| Optimizer | AdamW, betas 0.9/0.95 |
| Learning rate | 1e-3, 100-step warmup, cosine decay |
| Weight decay | 0.1 |
| Dropout | 0.1 |
| Hardware | 1 × NVIDIA T4 (Modal) |
| Seeds | 1 |

Hybrid model details:

- 4 unique transformer blocks, each applied for 2 loops (8 block applications/token).
- Per-loop, per-block zero-initialized bottleneck adapters, rank 16.
- Grouped-query causal attention: 4 query heads, 2 KV heads.
- Rotary position embeddings; no absolute position table.
- 4 or 8 experts per block, top-1 routing.
- Load-balancing auxiliary loss 0.01; router z-loss 0.001.
- Every loop produces logits and is trained with cross-entropy; exit decisions use
  mean maximum softmax probability.

### Full-depth results

Validation loss from 50 batches (32 sequences each), 1,600 validation windows.

| Model | Total params | Unique active params | Applications/token | Validation loss |
|---|---:|---:|---:|---:|
| Dense 8 layers | 1,619,968 | 1,619,968 | 1,619,968 | 1.1806 |
| Hybrid, 4 experts | 2,380,544 | 800,000 | 1,528,064 | 1.1364 |
| Hybrid, 8 experts | 4,489,984 | 802,048 | 1,532,160 | **1.0962** |

"Applications/token" counts parameter uses per token, including shared blocks once per
loop. It is a compute proxy, not measured FLOPs.

Per-loop validation losses for the 8-expert model: loop 1 = 1.1038, loop 2 = 1.0962.
The second loop gains 0.0076. For the 4-expert model: loop 1 = 1.1450, loop 2 = 1.1364,
a gain of 0.0086.

### Adaptive exits

Thresholds are applied to the mean maximum softmax probability over a validation window.
Losses below are on the same 1,600 calibration windows as the full-depth reference.

| Model | Threshold | Mean loops | Windows exiting after loop 1 | Validation loss |
|---|---:|---:|---:|---:|
| 4 experts | full | 2.00 | 0% | 1.1438 |
| 4 experts | 0.2 | 1.00 | 100% | 1.1527 |
| 4 experts | 0.6 | 1.08 | 91.5% | 1.1513 |
| 4 experts | 0.7 | 1.53 | 47% | 1.1481 |
| 8 experts | full | 2.00 | 0% | 1.1122 |
| 8 experts | 0.2 | 1.00 | 100% | 1.1195 |
| 8 experts | 0.6 | 1.07 | 92.9% | 1.1189 |
| 8 experts | 0.7 | 1.50 | 49.7% | 1.1159 |

The 8-expert model gives up 0.0073 validation loss to halve its average loop count.
A threshold of 0.7 gives a smoother compute-quality trade-off with about 50% of
windows exiting early.

### KV cache

Cached and uncached greedy/temperature sampling for 64 new tokens on T4.

| Model | Cached tok/s | Uncached tok/s | Speedup |
|---|---:|---:|---:|
| Dense 8 layers | 87.0 | 84.6 | 1.03× |
| Hybrid, 4 experts | 67.9 | 53.7 | 1.26× |
| Hybrid, 8 experts | 62.1 | 40.5 | 1.53× |

MoE routing overhead is visible in raw decode throughput. The cache avoids recomputing
the full shown context at every step; the relative gain grows with context length and
loop count. Adaptive exits and caching are used together only at prompt-level depth:
the depth chosen after the prompt is frozen for the sequence, keeping each loop's KV
cache valid.

### Generated sample

Prompt `def `, 8-expert model, exit threshold 0.2 (one loop):

```python
def _sequence(self, self, name=_name, number=True):
        """
        """
        The create a self    # a set as doess in error of an events also the True in the module line line.
        self._callbac
```

This is character-level output after roughly one epoch. Syntax is learned; semantics
are weak. `results/samples.md` contains a longer sample.

## Study 1: MLP depth on a toy task

3-class spiral, width 64, 100 epochs, AdamW, 5 seeds. `depth_experiment.py`.

| Hidden layers | Parameters | Validation accuracy |
|---:|---:|---:|
| 1 | 387 | 79.2% |
| 2 | 4,547 | 99.0% |
| 3 | 8,707 | 98.8% |
| 4 | 12,867 | 98.7% |
| 5 | 17,027 | 98.7% |

One hidden layer underfits; two reach the task ceiling; more layers add parameters
without measurable accuracy. This motivated treating the depth axis as something
to spend adaptively rather than assume more is better.

## Study 2: matrix-shaped layers

8×8 hidden grid, 3-class spiral, 3 seeds, 80 epochs. `matrix_experiment.py`.

| Layout | Parameters | Validation accuracy | Connections |
|---|---:|---:|---|
| Dense flat | 8,707 | 99.1% | every cell to every cell |
| Local matrix | 417 | 99.0% | 3×3 neighborhood, shared kernel |
| Row + column | 771 | 99.0% | two 8×8 factors per block |

Reshaping a vector into a matrix changes nothing by itself. Restricting edges and
sharing weights cut parameters by 21× at the same accuracy. The useful object is
the connection structure, not the tensor shape.

## Study 3: translation generalization

8×8 images containing a horizontal or vertical 1×3 bar, pixel noise 0.12.
Train on top-half bar positions; test on seen top-half and held-out bottom-half
positions. 768 training examples, 3 seeds. `spatial_test.py`.

| Model | Parameters | Seen positions | Held-out positions |
|---|---:|---:|---:|
| Dense MLP | 8,450 | 99.7% | 53.2% |
| Local 3×3 grid + pooling | 206 | 100% | 99.2% |

The dense model memorizes positions. Shared local filters plus global pooling carry
the task across unseen positions with 41× fewer parameters.

## Study 4: positional information in transformers

Same bar task, 64 per-pixel tokens, two attention blocks, mean pooling, 768 examples,
3 seeds. `transformer_test.py`.

| Model | Parameters | Seen positions | Held-out positions |
|---|---:|---:|---:|
| Local CNN reference | 206 | 100% | 99.2% |
| Transformer, relative 2D position bias | 5,446 | 100% | 97.8% |
| Transformer, absolute learned positions | 5,570 | 99.7% | 49.9% |
| Transformer, no position information | 4,546 | 49.9% | 50.2% |

Self-attention without position cannot distinguish horizontal from vertical bars
because the token multiset is the same. Absolute position tables memorize training
positions. Relative position bias preserves the geometry and transfers. This is the
same lesson used by the loop-level design: make structure relative, not location-bound.

## Study 5: character-level baseline language model

Tiny Shakespeare, 1.1M characters, 65-token character vocabulary, 4 layers, 4 heads,
128-dimensional embeddings, 2,000 steps. Validation loss 1.57. `language_model.py`.
This established the tokenizer, causal attention, RoPE, sampling, and checkpoint
pipeline used by the hybrid model.

## Threats to validity

- **One seed** for the main sweep. The 8-expert gain (0.040) is much larger than the
  within-model noise seen in other experiments, but a multi-seed run is required.
- **One epoch.** The code model saw 12.3M tokens from an 11.1M-character corpus.
  More data or longer training can change the expert-count conclusion.
- **Character-level tokenization.** BPE or a code-aware tokenizer may change both
  quality and the value of extra capacity.
- **Compute proxy.** Parameter applications per token ignore attention sequence cost,
  kernel efficiency, and hardware utilization. Wall-clock and memory are reported
  separately where measured.
- **No downstream evaluation.** Loss is the only metric. Code execution, unit tests,
  or human preference are more meaningful for code models.
- **Single cloud hardware target.** T4 timings do not transfer directly to other GPUs.

## Reproduction

```bash
python scripts/build_code_corpus.py
python hybrid_lm.py --name hybrid_e8 --experts 8 --steps 1500

# cloud sweep
python -m modal run modal_run.py
python -m modal volume get --force hybrid-lm-code-results hybrid_e8.pt .
python evaluate_hybrids.py --checkpoints dense8=modal_dense8.pt hybrid_e4=modal_hybrid_e4.pt hybrid_e8=modal_hybrid_e8.pt
python plot_modal_sweep.py
```

The figures in `results/` were produced by `plot_modal_sweep.py` from
`results/final_eval.json` and `results/modal_results.json`.
