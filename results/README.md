# Results

Artifacts produced by the experiments described in `README.md` and `RESULTS.md`.

## Main sweep

- `modal_sweep.png`, `modal_sweep.svg` — validation loss per active compute for the
  dense 8-layer, 4-expert hybrid, and 8-expert hybrid; adaptive exit curve for the
  8-expert model.
- `modal_results.json` — Modal training metrics, including total/active parameters,
  training throughput, per-loop validation losses, exit calibration during training,
  and T4 decoding speeds.
- `final_eval.json` — 50-batch re-evaluation of the saved checkpoints, including
  corrected exit curves, selected aggressive/balanced thresholds, generated samples,
  and local cached/uncached decoding benchmarks.

Validation loss in the figure and tables comes from `final_eval.json`, because the
10-batch evaluation during training was noisy. Hyperparameters and data are in
`RESULTS.md`.

## Auxiliary studies

- `mlp_architecture.png`, `mlp_architecture.svg` — configurable MLP diagram.
- `mlp_architecture_4_layers.png`, `mlp_architecture_4_layers.svg` — depth comparison.
- `depth_effect.png`, `depth_effect.svg` — MLP accuracy and parameters vs depth.
- `matrix_mlp.png`, `matrix_mlp.svg` — 8×8 hidden grids with dense, local, and
  row/column connection patterns.
- `spatial_test.png`, `spatial_test.svg` — translation generalization, dense vs
  local grid.
- `transformer_test.png`, `transformer_test.svg` — absolute, relative, and no
  positional information under position shift.

## Samples

`results/samples.md` contains a generated code sample. It is synthetic model output,
not real project source.

## Checkpoints

The trained `.pt` files are excluded from Git. Download them from the Modal volume:

```bash
python -m modal volume get --force hybrid-lm-code-results dense8.pt .
python -m modal volume get --force hybrid-lm-code-results hybrid_e4.pt .
python -m modal volume get --force hybrid-lm-code-results hybrid_e8.pt .
```

Then rerun `evaluate_hybrids.py`. The volume also contains `summary.json` and the
per-run metrics JSON.
