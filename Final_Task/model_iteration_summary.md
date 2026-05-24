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

## Recent Output Evidence

Old non-final Task 1 trained-run directories were removed because they are not comparable to final Network 5 / DIV 40 training. Their key metrics are preserved in:

`task1_outputs/old_task1_runs_summary.md`

Historical conclusions from those runs:

- best old validation accuracy was `43.29%` from `graph_temporal_proto_bit_cnn`
- the single-network run peaked early and then overfit
- the later multi-network DIV21 loop-topology run peaked lower at `36.41%`
- pattern IDs are nominal; class-circle assumptions should not drive the Task 1 loss or labels
- loop/topology information should be used only for electrode and response structure

The conclusion is that final Task 1 should be judged on the new shared `N5_DIV40.h5` run, not on old group-data or DIV21 outputs. Task 1 now uses validation macro-F1 for scheduler, early stopping, and checkpoint selection; accuracy is retained as a secondary diagnostic.

## Current Task 1 Architecture

The current Task 1 model is a graph-temporal classifier running on the final shared Network 5 / DIV 40 data. The code remains multi-network-capable, but the active final configuration intentionally uses only Network 5.

High-level idea:

`response tensor -> temporal electrode features -> spatial graph aggregation -> pooled trial representation -> pattern class logits`

The model predicts one of the 16 stimulation patterns.

### Inputs

For each trial, the model uses:

- binned neural response tensor
- stimulation frequency
- network ID
- electrode mask for padded electrodes
- physical electrode coordinates when available
- electrode-level static features
- train-only per-network electrode activity rates

The architecture still supports padded multi-network inputs, but the final run has one active network with 105 electrodes. The mask remains useful as a compatibility and safety mechanism.

### Multi-network handling

The notebook loads only Network `5` at DIV `40` from the shared final-data folder. The model still carries a `network_id` embedding for compatibility with the earlier architecture, but in the final setup it has a single active network domain.

### Temporal response encoder

For each electrode, the response over time is passed through shared temporal convolutional layers. This produces an electrode-level temporal feature vector.

Why this matters:

- Task 1 is not only about total firing rate
- spike timing and response shape can disambiguate patterns
- shared temporal filters reduce overfitting compared with fully flattening the response

### Electrode features

Each electrode representation combines:

- temporal response features
- physical 2D coordinates
- electrode index embedding
- electrode activity/rate feature
- network embedding

This is meant to keep electrode identity and physical position available without forcing the model to memorize one fixed network layout.

### Graph block

The model builds a k-nearest-neighbor graph from electrode coordinates and applies graph-style message passing with batch matrix multiplication.

Why this matters:

- nearby electrodes tend to have correlated response structure
- stimulation patterns may create spatially organized response fields
- graph aggregation gives each electrode local spatial context before global pooling

### Masked pooling

After graph processing, the model pools electrode states with masked statistics:

- masked mean
- masked max
- masked standard deviation

The mask prevents padded electrodes from contributing. Using several pooling statistics gives the classifier access to both global activity and localized high-response events.

### Frequency and prototype features

Frequency features are concatenated into the trial-level classifier input. The model also contains train-derived network-specific class prototypes and adds prototype similarity as an auxiliary signal to the class logits.

Why this matters:

- frequency changes response strength and timing
- prototypes give a simple stabilizing reference for each class/network
- the learned classifier can still override the prototype signal

### Heads and losses

The model has:

- a 16-class pattern classification head
- an auxiliary bit head for the 4-bit pattern representation

Training uses:

- cross entropy with light label smoothing
- auxiliary bit consistency loss
- response augmentation with small noise/electrode/time dropout

The auxiliary bit head is not the final output; it is there to encourage the class representation to respect the binary pattern structure.

## Current Task 1 Evaluation

The notebook writes new runs as:

`task1_outputs/task1_multinet_<split_mode>_<timestamp>/`

Important files:

- `summary.json`
- `training_history.csv`
- `accuracy_by_pattern.csv`
- `accuracy_by_frequency.csv`
- `accuracy_by_network.csv`
- `accuracy_by_network_pattern.csv`
- `confusion_matrix.csv`
- `best_validation_predictions.csv`
- `best_validation_arrays.npz`
- `best_model_this_run.pth`

It also saves the global best checkpoint to:

`/home/bnn_10fs26/best_model_final_task_1_multinetwork_graph_proto_bits.pth`

## Task 1 Splits

Default split:

`SPLIT_MODE = "random"`

This estimates validation performance on held-out trials from the final shared Network 5 / DIV 40 training file. `leave_network_out` is no longer meaningful for the active final setup because only Network 5 is used.

## Task 1 Conclusions So Far

What still looks useful for the final run:

- physical electrode coordinates instead of pure flattening
- electrode graph aggregation over neighboring/topological relations
- masked mean/max/std pooling
- light response regularization and early stopping
- nominal 16-class pattern labels without circular class smoothing

What old runs warned against:

- simply training longer after validation plateaus
- relying on multi-network DIV21 transfer as if it were comparable to final data
- treating pattern IDs as adjacent classes on a pattern circle
- reading high training accuracy as progress when validation accuracy is flat

The next decisive result is the first completed final Network 5 / DIV 40 Task 1 run.

# Task 2

## Current Notebooks

Current best stable notebook:

`task2_best_model.ipynb`

Current best worker:

`task2_best_model_worker.py`

Spike-localization exploration notebook:

`task2_hybrid_spike_localization_model.ipynb`

Current best model name:

`task2_best_C4_bits_H24_GRU2_freqRBF16`

## Task 2 Output Structure

Task 2 outputs now live under:

`task2_outputs/`

Current final-data output folder:

`task2_outputs/hybrid_spike_localization/task2_hybrid_gated_topology_N5_DIV40_shared_final_random/`

Retention layout:

- `current_run/` stores the latest run
- `previous_run/` stores the immediately preceding run after the next training run
- `best_run/` stores the best retained run by `final_metric_proxy_weighted_w1_ms`


Latest completed hybrid run:

- raw model weighted W1 proxy: `0.4138 ms`
- pattern-baseline weighted W1 proxy: `0.5241 ms`
- improvement over pattern baseline: `21.0%`
- count correlation: `0.9237` model vs `0.8679` pattern baseline
- raw model wins on `403/537` pattern-frequency cells, so remaining errors are concentrated condition-level failures rather than a global model failure

The notebook is now W1-centered. New runs write a validation-tuned model/pattern-baseline blend (`w1_blend_*.csv`) and ranked condition-level W1 diagnostics (`condition_w1_*.csv` plus heatmaps). The blend is deliberately post-hoc and validation-selected: it lets the submitted predictor fall back to the pattern baseline only in cells where validation W1 says the baseline is safer.

Old DIV21 Task 2 outputs were removed because they are not comparable to final data. Their key metrics are preserved in:

`task2_outputs/old_div21_runs_summary.md`

## Task 2 Objective

Task 2 predicts the full neural response distribution for a stimulation trial:

`[electrodes, time_bins]`

The TA feedback suggested that final evaluation is unlikely to be exact-response BCE only. It likely cares about distributional properties such as:

- whether firing rate matches the stimulus
- whether first-spike timing matches
- whether the predicted response distribution is plausible

This is why the later Task 2 work added count, PSTH, top-activity, and spike-localization diagnostics in addition to BCE.

## Current Task 2 Architecture

The current best Task 2 model is a coordinate-aware graph temporal residual decoder conditioned on current stimulus, frequency, and recent stimulation history.

High-level formula:

`logits = baseline_logits + residual_scale * learned_residual`

The model outputs spike probabilities for every electrode/time bin.

## Task 2 Inputs

For each trial, the model uses:

- current stimulation pattern represented as 4 current bits
- current frequency
- previous pattern ID
- causal z-window schedule summary
- 24-step previous-stimulation history
- electrode coordinates
- electrode IDs
- electrode-rate features
- fixed electrode graph adjacency

The target is the binned response tensor:

`[n_electrodes, n_time_bins]`

## Current Pattern Representation

The current pattern is represented by four binary bits:

`[(pattern >> 0) & 1, ..., (pattern >> 3) & 1]`

Those bits are passed through a linear projection:

`4 -> 16`

The current model deliberately does not use a learned current-pattern ID embedding. The reason is that a direct ID embedding can memorize pattern identity and may generalize poorly if the goal is to exploit pattern structure.

## Frequency Representation

The current model uses a compact direct frequency encoder with:

- normalized linear frequency
- normalized log frequency
- Fourier features on log-frequency
- 16 RBF features on log-frequency

Total direct frequency dimension:

`26`

More elaborate frequency machinery was tested and did not help enough to keep.

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

The best history encoder uses:

- history length: `24`
- history pattern mode: categorical embedding
- history pattern embedding dim: `16`
- continuous inputs per history step: `3`
- GRU hidden dim: `64`
- GRU layers: `2`

This was the most important model improvement in the final grid. The best run was the two-layer H24/D64 GRU variant, not the widest GRU and not the longest history window.

## Condition Vector

The condition vector concatenates:

- current bit projection: `16`
- frequency encoding: `26`
- previous pattern embedding: `16`
- causal z-window context: `28`
- z-history GRU state: `64`

Total before the condition MLP:

`150`

The condition MLP maps this to a `96`-dimensional condition vector.

## Electrode and Graph Decoder

Each electrode receives:

- physical coordinates
- normalized electrode index
- electrode-rate feature
- learned electrode embedding
- trial condition vector

The electrode state is processed by graph message passing over a k-nearest-neighbor electrode graph:

- `graph_k = 8`
- `n_graph_layers = 2`

The graph block helps the model use local spatial context instead of predicting each electrode independently.

## Temporal Decoder

The temporal decoder uses learned temporal basis functions. Each electrode predicts coefficients over temporal modes, and the response residual is reconstructed from those coefficients.

Current setting:

`k_modes = 80`

The temporal features include learned time embeddings plus normalized time, log-time, early-response features, and late-bin indicators.

## Task 2 Loss

The stable current model uses:

`BCE + 0.03 * count_loss + 0.03 * psth_loss`

BCE remains the main training objective. Count and PSTH losses are kept small because large distributional losses can improve aggregate-looking metrics while hurting precise spike-bin localization.

## What Worked in Task 2

The strongest retained ingredients were:

- current pattern as 4 bits with a linear projection
- 24-step stimulation history
- two-layer GRU with hidden size 64
- categorical embeddings for previous-pattern history tokens
- causal z-window context
- physical electrode coordinates
- graph message passing over electrodes
- residual decoding around a simple baseline
- small count/PSTH auxiliary losses
- RBF16 frequency features, but only as simple direct conditioning

The final history/frequency grid showed the best model was:

`hist_H24_D64_layers2`

with best validation BCE around:

`0.033972`

The one-layer C4 baseline in the same grid was around:

`0.034444`

So the biggest confirmed gain came from better history modeling.

## What Did Not Work in Task 2

The following did not justify keeping in the clean best model:

- frequency FiLM/gating
- larger RBF banks such as RBF32
- direct-frequency-only variants without z-window context
- using raw bits inside the history GRU
- very short history windows such as H8
- simply making the GRU wider without improving temporal depth
- multi-GPU data parallelism for one small model
- large spike-localization/ranking losses as direct replacements for BCE

The reason is mostly the same across these: they added complexity but did not consistently improve validation behavior. Many variants changed the model in several places at once or improved aggregate metrics while not clearly improving the hard spike-localization errors.

## Important Task 2 Caveat

The current Task 2 model still risks averaging too much because most bins are zero. BCE can look decent even if rare active bins are poorly localized. This is why future evaluations should always include:

- active-bin recall or top-k hit rate
- first-spike timing error
- count correlation and count MAE
- PSTH similarity
- high-activity pattern diagnostics
- per-electrode and per-time-bin error analysis

The current best C4 model is the stable baseline, but the hybrid spike-localization model remains relevant if the final evaluator rewards spike timing and active-region localization more than BCE.

## Current Task 2 Best Configuration

```text
model: task2_best_C4_bits_H24_GRU2_freqRBF16
current pattern: four bits, linear projection
current pattern ID embedding: disabled
frequency: normalized/log/Fourier/RBF16 direct encoder
frequency FiLM: disabled
context: causal z-window summary, 28 features
z-history length: 24
z-history pattern mode: categorical embedding
z-history GRU hidden dim: 64
z-history GRU layers: 2
graph: enabled, k=8, 2 graph layers
electrode embedding: enabled, dim=24
temporal modes: 80
condition dim: 96
hidden dim: 96
dropout: 0.12
weight decay: 1e-4
aux count weight: 0.03
aux PSTH weight: 0.03
batch size: 1024
max epochs: 160
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

Latest notebook improvement for the next run: the hybrid Task 2 model now uses a smoothed train-only pattern-frequency residual baseline (`baseline_mode="pattern_frequency"`, alpha `5.0`) plus categorical current-pattern embedding alongside the 4-bit features. This was added because the remaining W1 errors are pattern-frequency structured and pattern IDs are nominal. Future runs also use fixed `RANDOM_SEED=42` and save `split_indices.npz` for reproducible comparison.
