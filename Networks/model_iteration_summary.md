# Model Iteration Summary

Updated: 2026-05-27

## Scope

This repository contains the final notebooks, selected lightweight model outputs, and research notes for the two final tasks:

- `Pattern_Classification.ipynb` - Task 1 exact stimulation-pattern classification.
- `Spike_Response_Prediction.ipynb` - Task 2 electrode-by-time spike-response prediction.

The included output summaries correspond to the selected final validation and inference runs. Large generated Task 2 artifacts and model checkpoints are intentionally left out of git.

## Final Data Setup

Both notebooks use the shared final-data files provided for the project.

Task 1 configuration:

```python
Network = 6
DIV = 40
group_data = False
test_mode = False
```

Task 2 configuration:

```python
Network = 5
DIV = 40
group_data = False
test_mode = False
```

The corresponding `_test` files are used only for final inference because they omit the supervised target for the task being predicted.

## Output Directories

Task 1 selected output:

`task1_outputs/task1_N6_DIV40_macro_f1_seed42_20260525_024957/`

Task 2 selected output:

`task2_outputs/hybrid_spike_localization_grid/task2_hybrid_gated_topology_N5_DIV40_grid_robust_blend_tuning/grid_robust_blend_next_16_best_blend_seed7/`

# Task 1

## Notebook

`Pattern_Classification.ipynb`

The notebook trains the selected exact-pattern classifier for `N6_DIV40.h5` and writes final `_test` predictions with shape `(9839,)`.

## Selected Run

Run:

`task1_outputs/task1_N6_DIV40_macro_f1_seed42_20260525_024957/`

Checkpoint:

`/home/bnn_10fs26/best_model_pattern_classification_N6_DIV40_macro_f1.pth`

Key validation metrics:

| metric | value |
| --- | ---: |
| status | `early_stopped` |
| completed epochs | `45` |
| best epoch | `29` |
| validation macro-F1 | `98.5313%` |
| validation accuracy | `98.5288%` |
| validation weighted-F1 | `98.5226%` |
| validation micro-F1 | `98.5288%` |
| validation loss at best | `0.057720` |
| validation errors | `108 / 7341` |

Pattern `2` was the weakest validation class, with `85.0107%` accuracy and `91.2644%` F1. Most other patterns were at or above approximately `98.7%` accuracy.

The final inference cell completed successfully and wrote the Task 1 prediction HDF5 with `stimulation_patterns` shape `(9839,)`.

## Task 1 Architecture

Model type:

`task1_final_N6_DIV40_macro_f1_v1`

Model summary:

`response tensor -> raw electrode-time MLP branch + temporal graph branch + prototypes + class attention -> pattern logits`

The model predicts one of 16 nominal stimulation-pattern IDs. Pattern IDs are not treated as circular or ordinal labels.

Inputs used by the final configuration:

- binned neural response tensor
- network ID
- electrode mask for padded electrodes
- physical electrode coordinates when available
- electrode-level static features
- train-only per-network electrode activity rates

Frequency features are supported by the module but disabled in the final Task 1 configuration:

```python
USE_FREQUENCY_FEATURES = False
```

The raw electrode-time branch uses adaptive electrode-time windows, electrode/time activity marginals, total activity, LayerNorm, an MLP, and additive class logits. The graph branch adds electrode topology context through kNN/ring/self relations.

Training defaults:

```python
NETWORK_CANDIDATES = [6]
DIV = 40
VAL_FRAC = 0.10
RANDOM_SEED = 42
n_epochs = 100
batch_size = 256
learning_rate = 8e-4
weight_decay = 1e-4
MODEL_DROPOUT = 0.14
RAW_TIME_WINDOWS = 8
CLASS_LABEL_SMOOTHING = 0.0
EARLY_STOP_PATIENCE = 16
EARLY_STOP_MIN_DELTA = 1e-4
```

# Task 2

## Notebook

`Spike_Response_Prediction.ipynb`

The notebook defines the final Task 2 model family for `N5_DIV40.h5` and performs final `_test` inference from the selected checkpoint:

`task2_outputs/hybrid_spike_localization_grid/task2_hybrid_gated_topology_N5_DIV40_grid_robust_blend_tuning/grid_robust_blend_next_16_best_blend_seed7/best_model_this_variant.pth`

The inference cell applies the validation-selected W1 blend tables when available and writes the Task 2 prediction HDF5 with `binned_spike_train_responses` shape `(9839, 105, 80)`.

## Selected Run

Run:

`task2_outputs/hybrid_spike_localization_grid/task2_hybrid_gated_topology_N5_DIV40_grid_robust_blend_tuning/grid_robust_blend_next_16_best_blend_seed7/`

Key metrics:

| metric | value |
| --- | ---: |
| final blended weighted W1 proxy | `0.3506 ms` |
| raw model weighted W1 proxy | `0.4569 ms` |
| pattern baseline weighted W1 proxy | `0.5141 ms` |
| pattern-frequency baseline weighted W1 proxy | `0.4392 ms` |
| blend improvement over raw model | `23.3%` |
| blend improvement over pattern baseline | `31.8%` |
| nested held-out blend W1 | `0.5433 ms` |
| nested held-out raw-model W1 | `0.5578 ms` |
| best epoch | `156` |
| epochs run | `176` |

This checkpoint is used by `Spike_Response_Prediction.ipynb` for final test inference.

## Task 2 Architecture

The selected Task 2 family predicts the full neural response distribution:

`[electrodes, time_bins]`

The model is a coordinate-aware graph temporal residual decoder conditioned on current stimulus, frequency, and causal stimulation history.

Model form:

`logits = baseline_logits + residual_scale * learned_residual`

Core components:

- train-only pattern-frequency residual baseline
- categorical current-pattern embedding plus compact bit features
- direct frequency features with RBF encoding
- 24-step stimulation-history GRU
- 28-dimensional causal z-window context
- physical electrode coordinates and electrode-rate features
- gated hybrid electrode graph relations: kNN, soft loop/topology, response correlation, and self
- hard zero mask for globally silent early bins `0-5`
- validation-selected W1 model/baseline blending using pattern and pattern-frequency anchors

Selected run defaults:

```text
variant: grid_robust_blend_next_16_best_blend_seed7
training data: shared final N5_DIV40.h5 only
test file: N5_DIV40_test.h5 for inference only
split: seed_7
validation fraction: 0.25
random seed: 42
checkpoint metric: final W1-proxy blend
baseline mode: pattern_frequency
pattern-frequency baseline alpha: 5.0
pattern-frequency activity alpha boost: 1.0
pattern-frequency activity alpha power: 1.0
W1 blend anchors: pattern + pattern_frequency
W1 blend min trials: 8
electrode-level W1 blend: enabled
electrode-level min true mass: 1e-4
history length: 24
history GRU hidden dim: 64
history GRU layers: 2
frequency RBF dim: 16
graph: gated hybrid kNN + soft_loop + response_corr + self
dropout: 0.12
optimizer: AdamW
learning rate: 8e-4
weight decay: 1e-4
aux count weight: 0.03
aux PSTH weight: 0.03
temporal W1 schedule: 0.04 -> 0.08 -> 0.12
```

## Task 2 Notes

Model selection is W1-centered. BCE and count calibration are included as diagnostics, while final selection is based on weighted PSTH W1 and condition-level blend diagnostics.

Pattern IDs are nominal stimulation IDs. Electrode circular/topology features describe output electrode layout only; they are not used to impose circular order on stimulation-pattern IDs.
