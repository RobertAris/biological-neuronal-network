# Model Iteration Summary

This document summarizes the current state of Task 1 and Task 2 after the recent cleanup. It records what worked, what did not, where the outputs live, and how the current model architectures are structured.

## Output Structure

Task 1 outputs are stored in:

`task1_outputs/`

Task 2 outputs are stored in:

`task2_outputs/`

Both folders now contain a `README.md` explaining their internal structure. Old non-final trained outputs were removed after their key metrics were consolidated into summary files.

## Current Final-Data Configuration

All runnable training notebooks and helper scripts now default to the shared final dataset:

```python
Network = 5
DIV = 40
group_data = False
test_mode = False
```

`N5_DIV40_test.h5` is treated as inference/submission data only. It is not used for supervised training because it omits spike responses. Historical DIV21/non-final output directories have been removed; their key results are retained in `task1_outputs/old_task1_runs_summary.md` and `task2_outputs/old_div21_runs_summary.md`.

# Task 1

## Current Notebook

Current cleaned notebook:

`Final_Task_1.ipynb`

Current goal:

- classify the stimulation pattern from neural response data
- optimize validation macro-F1 as the primary metric from the final-data PDF, while tracking accuracy as a secondary diagnostic
- train/evaluate on the final shared Network 5 / DIV 40 dataset only

## Current Status

Task 1 has moved from the old ~43% historical result to a near-saturated final-data model on the shared `N5_DIV40.h5` training file.

Current best completed run:

`task1_outputs/task1_N5_DIV40_macro_f1_random_20260524_194723/`

Key metrics from that run:

- validation macro-F1: `99.9613%`
- validation accuracy: `99.9609%`
- best epoch: `29`
- validation errors: `3 / 7674`
- Ridge electrode-time baseline macro-F1: `97.4518%`
- Ridge electrode-time baseline accuracy: `97.4590%`

The neural model fixed `192 / 195` Ridge validation mistakes and introduced no new mistakes relative to Ridge on the seed-42 validation split. The remaining mistakes were:

| frequency | true pattern | predicted pattern |
| ---: | ---: | ---: |
| 3 Hz | 5 | 1 |
| 12 Hz | 1 | 5 |
| 38 Hz | 7 | 3 |

A concise retained run summary is stored in:

`task1_outputs/task1_final_runs_summary.md`

## Current Task 1 Architecture

The current Task 1 model is a hybrid raw-electrode-time + graph-temporal classifier. The main lesson from the baseline analysis was that the final data are highly separable in coarse electrode-time response space, so the model now preserves that direct signal instead of forcing all information through graph pooling.

High-level idea:

`response tensor -> raw electrode-time MLP branch + temporal graph branch + prototypes + class attention -> pattern logits`

The model predicts one of the 16 stimulation patterns.

### Inputs

For each trial, the model uses:

- binned neural response tensor
- network ID
- electrode mask for padded electrodes
- physical electrode coordinates when available
- electrode-level static features
- train-only per-network electrode activity rates

Frequency features are supported by the code but are disabled by default:

```python
USE_FREQUENCY_FEATURES = False
```

The response tensor alone was enough for strong validation performance, and disabling frequency keeps Task 1 inference less dependent on metadata.

### Raw Electrode-Time Branch

The direct branch uses coarse response features that mirror the strong Ridge baseline:

- adaptive electrode-time windows, currently `RAW_TIME_WINDOWS = 8`
- electrode activity marginals
- time activity marginals
- total activity
- LayerNorm + MLP + additive class logits

This branch is the main protection against losing electrode-time identity.

### Temporal Graph Branch

The graph path is retained as an auxiliary learned representation:

- shared temporal convolutions per electrode
- electrode coordinate and identity features
- train-only electrode activity/impedance/ring-index static features
- multi-relation graph message passing over kNN/ring/self relations
- masked mean/max/std pooling

The graph branch can add spatial context, but it is no longer the only path to the classifier.

### Class Attention And Prototypes

The model also includes:

- class-specific attention pooling over electrode states
- train-derived class prototypes in raw response space
- learnable scales for raw branch, class attention, and prototypes

In the best checkpoint, the learned scales were roughly:

- raw branch scale: `1.10`
- prototype scale: `1.09`
- class-attention scale: `0.40`

### Training Setup

Current defaults:

```python
SPLIT_MODE = "random"
VAL_FRAC = 0.10
RANDOM_SEED = 1337
n_epochs = 100
batch_size = 256
learning_rate = 8e-4
weight_decay = 1e-4
MODEL_DROPOUT = 0.14
RESPONSE_NOISE_STD = 0.0
ELECTRODE_DROPOUT_PROB = 0.0
TIME_DROPOUT_PROB = 0.0
```

The old heavy regularization was removed because the final dataset is large and the discriminative response pattern is crisp.

## Current Task 1 Evaluation

The notebook writes new runs as:

`task1_outputs/task1_N5_DIV40_macro_f1_<split_tag>_<timestamp>/`

For random splits, the run name now includes the split seed, for example:

`task1_N5_DIV40_macro_f1_random_seed1337_<timestamp>/`

Important files:

- `summary.json`
- `training_history.csv`
- `ridge_baseline_summary.json`
- `ridge_baseline_validation_predictions.csv`
- `f1_summary.csv`
- `f1_by_pattern.csv`
- `accuracy_by_pattern.csv`
- `accuracy_by_frequency.csv`
- `accuracy_by_network.csv`
- `accuracy_by_network_pattern.csv`
- `confusion_matrix.csv`
- `best_validation_predictions.csv`
- `best_validation_arrays.npz`
- `best_model_this_run.pth`

It also saves the global best checkpoint to:

`/home/bnn_10fs26/best_model_final_task_1_N5_DIV40_macro_f1.pth`

## Task 1 Splits

Default split:

```python
SPLIT_MODE = "random"
RANDOM_SEED = 1337
```

The previous completed run used the original seed-42 split before seed metadata was added to run names. The seed-1337 split was smoke-tested with the Ridge baseline:

- Ridge validation accuracy: `97.6023%`
- Ridge validation macro-F1: `97.5937%`
- validation samples: `7674`
- overlap with old seed-42 validation set: `706 / 7674` samples (`9.20%`)

For an overfitting check, retrain from scratch on seed 1337. Do not only evaluate the seed-42 checkpoint on seed 1337, because about `90.80%` of seed-1337 validation samples were training samples in the seed-42 run.

## Task 1 Conclusions So Far

What worked:

- preserving coarse electrode-time identity directly
- using a Ridge baseline as a floor and sanity check
- adding a raw electrode-time MLP branch to the neural model
- keeping the graph/prototype path as an additive correction rather than a bottleneck
- disabling frequency features by default
- reducing regularization and response dropout

What old runs warned against:

- judging final Task 1 by old non-final/group-data outputs
- relying on graph pooling as the only route to class logits
- treating pattern IDs as adjacent classes on a pattern circle
- using heavy response/electrode/time dropout before the model beats the linear floor

Next useful check:

- run a full seed-1337 retrain from scratch and compare macro-F1/errors against the seed-42 result.

# Task 2

## Current Notebooks

Active final-data notebook:

`task2_hybrid_spike_localization_model.ipynb`

Grid-search notebook:

`task2_w1_grid_search.ipynb`

Baseline-compatible notebook:

`task2_best_model.ipynb`

Baseline worker:

`task2_best_model_worker.py`

Current active model name:

`task2_hybrid_gated_topology`

Historical stable C4 baseline name:

`task2_best_C4_bits_H24_GRU2_freqRBF16`

The C4 notebook remains useful as a sanity baseline, but the active final-data model is now the hybrid W1-centered spike-localization notebook.

## Task 2 Output Structure

Task 2 outputs now live under:

`task2_outputs/`

Current final-data output folder:

`task2_outputs/hybrid_spike_localization/task2_hybrid_gated_topology_N5_DIV40_shared_final_random/`

Retention layout:

- `current_run/` stores the latest run
- `previous_run/` stores the immediately preceding completed/attempted run
- `best_run/` stores the best retained run by `final_metric_proxy_weighted_w1_ms`

Old DIV21 Task 2 outputs were removed because they are not comparable to final Network 5 / DIV 40 training. Their key metrics are preserved in:

`task2_outputs/old_div21_runs_summary.md`

## Latest Completed/Retained Evidence

The best retained fully summarized hybrid run currently reports:

- final blended weighted W1 proxy: `0.4018 ms`
- raw model weighted W1 proxy: `0.4293 ms`
- pattern-only baseline weighted W1 proxy: `0.5203 ms`
- validation blend improvement over raw model: `6.4%`
- validation blend improvement over pattern baseline: `22.8%`
- best epoch: `96`
- checkpoint metric at that time: validation temporal-W1 proxy `0.06477`

A later fixed-seed run using the pattern-frequency baseline and categorical current-pattern embedding completed successfully but did not beat the retained best:

- final blended weighted W1 proxy: `0.4034 ms`
- raw model weighted W1 proxy: `0.4405 ms`
- pattern-frequency baseline weighted W1 proxy: `0.4466 ms`
- pattern-only baseline weighted W1 proxy: `0.5287 ms`
- best epoch: `18`

The raw residual model barely improved over the pattern-frequency baseline, which showed that most of the final gain came from validation blending. Pattern `0` was the clearest failure case: pattern-frequency W1 was worse than the simpler pattern-only baseline, and the residual gate was near its minimum, so the model mostly copied the wrong baseline for that quiet sparse pattern.

Earlier crash note, now fixed: a previous run reached the final evaluation stage but crashed in post-training W1 blend evaluation because `optimize_validation_w1_blend` referenced `blend_anchor_prob` inside the helper instead of its local `pattern_prob` argument. The helper now uses its local anchor argument correctly.

Partial evidence from that crashed run:

- best validation temporal-W1 proxy: `0.065516` at epoch `122`
- previous retained best validation temporal-W1 proxy: `0.064770`
- relation gates learned approximately `knn=0.979`, `soft_loop=0.666`, `response_corr=0.827`, `self=0.584`
- pattern-frequency baseline and categorical pattern embedding did not obviously improve the temporal-W1 proxy by themselves

Conclusion: the next run should not rely only on the better baseline. It should use a baseline that backs off for quiet patterns and select/check final predictions with the same kind of W1-aware model/baseline blending used after training.

## Current Task 2 Objective

Task 2 predicts the full neural response distribution for a stimulation trial:

`[electrodes, time_bins]`

The final-data PDF emphasizes normalized 1 ms PSTH Wasserstein-1 distance. The active notebook therefore treats weighted W1 as the primary final-style metric, while BCE, count calibration, and active-bin diagnostics remain guardrails.

Important metrics now written after a completed run include:

- raw model weighted W1 proxy
- blended weighted W1 proxy
- pattern baseline weighted W1 proxy
- pattern-frequency baseline weighted W1 proxy
- ranked pattern-frequency W1 table
- pattern-frequency W1 heatmap
- validation blend tables by condition, pattern, and condition-electrode
- distributional diagnostics such as count MAE/correlation and top-k spike capture

## Current Task 2 Architecture

The active model is a coordinate-aware graph temporal residual decoder conditioned on current stimulus, frequency, and recent stimulation history.

High-level formula:

`logits = baseline_logits + residual_scale * learned_residual`

The baseline is now a train-only pattern-frequency baseline with activity-adaptive smoothing rather than only a pattern baseline:

`baseline_mode = "pattern_frequency"`

`pattern_frequency_baseline_alpha = 5.0`

`pattern_frequency_activity_alpha_boost = 1.0`

This was chosen because validation probes showed the pattern-frequency baseline has much better W1 overall, but quiet patterns such as pattern `0` can get worse when split too finely by frequency. The adaptive pseudo-count pulls low-activity patterns back toward the pooled pattern-only template while keeping active patterns frequency-specific. The baseline is used as the residual anchor, not as a replacement for the neural model.

## Task 2 Inputs

For each trial, the active hybrid model uses:

- current stimulation pattern ID as a categorical embedding
- current stimulation pattern as four binary bits
- current frequency with normalized/log/Fourier/RBF16 features
- previous pattern ID
- causal z-window schedule summary
- 24-step previous-stimulation history
- electrode coordinates
- circular/topological electrode features
- electrode IDs
- electrode-rate features
- gated hybrid electrode graph relations

The target is the binned response tensor:

`[n_electrodes, n_time_bins]`

## Current Pattern Representation

Pattern IDs are nominal stimulation IDs. They are not treated as adjacent classes on a pattern circle.

The current active Task 2 model uses both:

- categorical current-pattern embedding
- 4-bit pattern features

The categorical embedding was restored because bit-only encoding was probably too restrictive for Task 2. The 4-bit features remain available as compact structure, but the model no longer has to pretend that bit similarity fully explains response similarity.

## Frequency Representation

The current model uses a compact direct frequency encoder with:

- normalized linear frequency
- normalized log frequency
- Fourier features on log-frequency
- 16 RBF features on log-frequency

More elaborate frequency machinery such as frequency FiLM/gating was tested earlier and did not help enough to keep.

## Causal Z-window Context

The model uses a 28-dimensional causal context vector derived only from previous stimulation schedule, not from the current response.

It includes:

- previous frequency
- time since previous stimulation
- rolling previous frequency over three steps
- current-minus-previous frequency
- window summaries over previous `3`, `5`, `10`, and `20` trials

For each window it includes frequency mean, frequency standard deviation, last frequency, current-minus-last frequency, pattern diversity, and same-current-pattern fraction.

This context stayed because it is cheap, causal, and useful.

## Z-history GRU

The history encoder uses:

- history length: `24`
- history pattern mode: categorical embedding
- history pattern embedding dim: `16`
- continuous inputs per history step: `3`
- GRU hidden dim: `64`
- GRU layers: `2`

This was the most important model improvement in the final grid. The best history setup was the two-layer H24/D64 GRU variant, not the widest GRU and not the longest history window.

## Electrode and Graph Decoder

Each electrode receives:

- physical coordinates
- normalized electrode index
- circular/topology features from the electrode layout
- electrode-rate feature
- learned electrode embedding
- trial condition vector

The graph decoder uses gated hybrid relations:

- physical kNN relation, `graph_k = 8`
- soft loop/topology relation, `loop_graph_k = 3`
- response-correlation relation, `response_graph_k = 8`
- self relation

The loop information is used only as an electrode/output topology prior. It is not used to impose circular order on pattern IDs.

## Temporal Decoder

The temporal decoder uses learned temporal basis functions. Each electrode predicts coefficients over temporal modes, and the response residual is reconstructed from those coefficients.

Current setting:

`k_modes = 80`

The temporal features include learned time embeddings plus normalized time, log-time, early-response features, and late-bin indicators. Globally impossible early bins `0-5` are hard-masked.

## Current Task 2 Loss and Checkpointing

The active run no longer trains with only BCE plus small count/PSTH auxiliaries. It now uses a W1 warmup schedule:

```text
loss = BCE + 0.03 * count_loss + 0.03 * PSTH_loss + temporal_w1_weight * temporal_W1_loss

epoch 1-20:  temporal_w1_weight = 0.04
epoch 21-60: temporal_w1_weight = 0.08
epoch 61+:   temporal_w1_weight = 0.12
```

BCE remains the calibration anchor. The W1 term becomes stronger later so the model first learns stable spike probabilities, then optimizes timing distribution more directly.

The optimizer is:

```python
AdamW(lr=8e-4, weight_decay=1e-4)
```

The scheduler is:

```python
ReduceLROnPlateau(mode="min", factor=0.5, patience=4)
```

Checkpoint selection and scheduler stepping now use a blend-aware final-style W1 proxy:

```python
CHECKPOINT_SELECTION_METRIC = "final_w1_proxy_blend_weighted_w1_ms"
```

This is a streaming validation proxy for the final weighted 1 ms PSTH W1 metric after conservative model/baseline blending. This matters because the shipped predictor is blended; selecting checkpoints only by raw-model W1 can stop too early or select a model whose residual is not useful after blending. Early stopping uses patience `10` validation checks with `EARLY_STOP_MIN_DELTA = 2e-5`.

## W1 Blend and Diagnostics

After training, the notebook fits a validation-selected multi-anchor blend:

`prediction = alpha * model + (1 - alpha) * baseline`

For the active run, validation can choose either of these anchors per pattern-frequency cell:

- pattern-only baseline
- activity-smoothed pattern-frequency baseline

The blend is selected by validation W1 at pattern-frequency level, with pattern-level fallback for low-trial cells and optional electrode-level refinement. This directly addresses the pattern `0` failure mode where pattern-frequency was worse than pattern-only.

The previous crash in this path has been fixed: the electrode-level helper now uses the local `pattern_prob`/anchor argument rather than an outer-scope `blend_anchor_prob` variable.

Diagnostics to inspect after each run:

- `psth_wasserstein_metrics.csv`
- `condition_w1_ranked.csv`
- `w1_blend_summary.csv`
- `w1_blend_by_condition.csv`
- `w1_blend_by_condition_electrode.csv`
- `plots/w1_by_pattern_freq.png`
- pattern-frequency baseline panel inside `plots/w1_by_pattern_freq.png`
- `plots/condition_w1_blend_heatmap.png`
- `plots/condition_w1_model_minus_pattern_heatmap.png`

## What Worked in Task 2

The strongest retained ingredients are:

- final-data-only training on Network 5 / DIV 40
- fixed `RANDOM_SEED = 42` and saved `split_indices.npz`
- current pattern as nominal categorical ID plus 4-bit features
- activity-adaptively smoothed train-only pattern-frequency residual baseline
- 24-step stimulation history
- two-layer GRU with hidden size 64
- categorical embeddings for history pattern tokens
- causal z-window context
- physical electrode coordinates
- graph message passing over electrodes
- gated kNN/soft-loop/response-correlation topology relations
- residual decoding around a baseline
- blend-aware W1 checkpoint selection and condition-level diagnostics

## What Did Not Work or Needs Caution in Task 2

The following did not justify becoming the main next change:

- treating pattern IDs as adjacent/circular classes
- relying on pattern-frequency baseline alone without more W1 pressure
- using only one blend anchor when pattern-only is safer for sparse quiet cells
- frequency FiLM/gating
- larger RBF banks such as RBF32
- direct-frequency-only variants without z-window context
- using raw bits inside the history GRU
- very short history windows such as H8
- simply making the GRU wider without improving temporal depth
- large focal/ranking losses as direct replacements for BCE

The main caution remains that BCE can look acceptable while final W1 or active-bin localization is still weak. W1, condition-level heatmaps, and top-k active-bin diagnostics should drive decisions.

## Current Task 2 Active Configuration

```text
model: task2_hybrid_gated_topology
training data: shared final N5_DIV40.h5 only
test file: N5_DIV40_test.h5 for inference/submission reference only
current pattern ID embedding: enabled
current pattern bits: enabled, linear projection
baseline mode: pattern_frequency
pattern-frequency baseline alpha: 5.0
pattern-frequency activity alpha boost: 1.0
W1 validation blend anchors: pattern + pattern_frequency
frequency: normalized/log/Fourier/RBF16 direct encoder
frequency FiLM: disabled
context: causal z-window summary, 28 features
z-history length: 24
z-history pattern mode: categorical embedding
z-history GRU hidden dim: 64
z-history GRU layers: 2
graph: gated hybrid kNN + soft_loop + response_corr + self
graph k: 8
loop graph k: 3
response graph k: 8
electrode embedding: enabled, dim=24
temporal modes: 80
condition dim: 96
hidden dim: 96
dropout: 0.12
optimizer: AdamW
learning rate: 8e-4
scheduler: ReduceLROnPlateau, factor 0.5, patience 4
weight decay: 1e-4
aux count weight: 0.03
aux PSTH weight: 0.03
temporal W1 schedule: 0.04 -> 0.08 -> 0.12
checkpoint metric: final_w1_proxy_blend_weighted_w1_ms
batch size: 1024
max epochs: 180
early stopping patience: 10 validation checks
```

# Big-picture Recommendations

## Task 1

First priority is to train the final Network 5 / DIV 40 run and compare its validation macro-F1, accuracy, and confusion structure against the old historical evidence. Do not mix in old DIV21 group data for final training.

If it still does not exceed `50%`, the next model change should focus on better discriminative supervision:

- stronger class-balanced loss
- hard-confusion mining for commonly confused patterns
- contrastive pattern representation
- spike-localization features before classification
- leave-network-out validation to detect network memorization

## Task 2

Use `task2_hybrid_spike_localization_model.ipynb` as the active final-data model. `task2_best_model.ipynb` remains a baseline-compatible notebook and now also defaults to N5/DIV40 shared data.

The most promising future direction is W1-centered calibration and condition-level validation, not another small BCE improvement:

- use weighted 1 ms PSTH W1 as the primary final-style metric
- keep BCE, count, and active-bin diagnostics as guardrails
- inspect ranked pattern-frequency cells instead of only marginal pattern/frequency plots
- use validation-selected model/pattern-baseline blending where it improves W1
- avoid losses that improve smooth averages but damage active-bin precision

Latest active notebook improvement: the hybrid Task 2 model now uses an activity-adaptively smoothed train-only pattern-frequency residual baseline (`baseline_mode="pattern_frequency"`, alpha `5.0`, low-activity boost `1.0`) plus categorical current-pattern embedding alongside the 4-bit features. It also trains with a temporal-W1 schedule (`0.04 -> 0.08 -> 0.12`), selects checkpoints with a blend-aware streaming final-style W1 proxy (`final_w1_proxy_blend_weighted_w1_ms`), and fits a multi-anchor validation blend that can fall back to either pattern-only or pattern-frequency baselines per condition. Runs use fixed `RANDOM_SEED=42` and save `split_indices.npz` for reproducible comparison.

A dedicated grid-search launcher, `task2_w1_grid_search.ipynb`, now supports staged W1-centered sweeps rather than one giant grid. The previous `targeted_next_12` stage found the best current validation result in `grid_targeted_next_12_blend_min8_mass1e4`: blended weighted W1 `0.3641`, raw model W1 `0.4582`, pattern-frequency baseline W1 `0.4464`. The active Task2 notebook now promotes that best proven blend setting (`W1_BLEND_MIN_TRIALS=8`, `W1_BLEND_ELECTRODE_MIN_TRUE_MASS=1e-4`). The new default grid stage is `robust_blend_next_16`: sixteen high-information follow-up runs that rerun the winner, probe nearby blend sensitivity, test validation split seeds `7`, `123`, and `2026`, combine the winning blend with the best baseline/W1-schedule candidates, and enable nested held-out blend evaluation. By default the notebook launches four parallel worker processes through `task2_w1_grid_worker.py`; each worker is pinned to one GPU, uses `REQUESTED_NUM_GPUS=1`, and trains a non-overlapping slice of the configs. New outputs go under `task2_outputs/hybrid_spike_localization_grid/task2_hybrid_gated_topology_N5_DIV40_grid_robust_blend_tuning/`, leaving the retained current/previous/best run folders and the previous `staged_w1_tuning` grid untouched.

Partial robust-grid follow-up results: the run was interrupted before all 16 configs completed, but four configs finished cleanly. Among completed runs, `grid_robust_blend_next_16_pfA8_boost2_pow075_bestblend` was best on both normal blended W1 (`0.35947`) and nested held-out blend W1 (`0.54868`). It also slightly improved raw model W1 (`0.45597` vs `0.45817`). Based on that signal, the active Task2 defaults now use `pattern_frequency_baseline_alpha=8.0`, `pattern_frequency_activity_alpha_boost=2.0`, and `pattern_frequency_activity_alpha_power=0.75`, while retaining the previously best blend settings (`W1_BLEND_MIN_TRIALS=8`, electrode-level blending on, `W1_BLEND_ELECTRODE_MIN_TRUE_MASS=1e-4`). The no-electrode blend probe was not promoted because it degraded normal final W1 to about `0.4004`.

