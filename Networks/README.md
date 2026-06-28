# Final Experiments

This directory contains the two final analyses for in vitro neural response modeling. The notebooks are written as executable experiment records: they define the model, load the shared data, train or restore the selected configuration, report validation diagnostics, and write final `_test` predictions.

## Entry Points

| notebook | focus | final data |
| --- | --- | --- |
| `Pattern_Classification.ipynb` | stimulation-pattern classification | `N6_DIV40` |
| `Spike_Response_Prediction.ipynb` | electrode-by-time spike-response prediction | `N5_DIV40` |

`model_iteration_summary.md` is the compact research log for the final selected configurations, including architecture notes, checkpoints, validation splits, and metrics.

## Selected Results

| task | selected run | headline result |
| --- | --- | ---: |
| Task 1 | `task1_outputs/task1_N6_DIV40_macro_f1_seed42_20260525_024957/` | `98.5313%` validation macro-F1 |
| Task 2 | `grid_robust_blend_next_16_best_blend_seed7` | `0.3506 ms` blended weighted W1 proxy |

Task 1 predicts one of 16 nominal stimulation-pattern IDs from the response tensor. The selected classifier reached `98.5288%` validation accuracy and produced final test predictions with `stimulation_patterns` shape `(9839,)`.

Task 2 predicts the full response tensor `[trials, electrodes, time_bins]`. The selected model combines a graph temporal residual decoder with validation-selected W1 blending and produced final test predictions with `binned_spike_train_responses` shape `(9839, 105, 80)`.

## Artifacts

- `task1_outputs/` contains lightweight validation summaries, CSV diagnostics, and the selected Task 1 run metadata.
- `task2_outputs/` is reserved for generated Task 2 runs. The heavy generated outputs and checkpoints are ignored by git; see `task2_outputs/README.md` for the selected run summary.
- `utils/` contains the shared data loading, saving, and plotting helpers used by both notebooks.

Model checkpoints (`*.pth`) are not committed. The selected checkpoint paths from the original training environment are documented in `model_iteration_summary.md`.

## Reproduction Notes

Both notebooks load data through `utils.data.load_data` and write final inference files through `utils.data.save_data`. The shared `_test` files are used only for final inference because they omit the supervised target being predicted.

The final configurations use:

- Task 1: `Network = 6`, `DIV = 40`, stratified random validation split, seed `42`.
- Task 2: `Network = 5`, `DIV = 40`, random validation split, validation fraction `0.25`, seed `42`.
