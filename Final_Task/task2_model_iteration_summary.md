# Task 2 Model Iteration Summary

Current best notebook: `task2_best_model.ipynb`

Current best model name in the cleaned version: `task2_best_C4_bits_H24_GRU2_freqRBF16`

This document summarizes the Task 2 modeling path, what mattered, what did not, and why. It combines the retained workspace outputs with the conclusions from the iteration chat.

## Evidence Sources

The most relevant retained files are:

- `task2_best_model.ipynb`: current cleaned notebook, now containing only the single best model.
- `task2_best_model_worker.py`: standalone worker generated from the cleaned notebook.
- `task2_assumption_audit_outputs/task2_assumption_audit_summary.json`: data and baseline audit.
- `task2_archive_summary/current_bits_baseline/`: archived clean current-bits baseline run summaries.
- `task2_archive_summary/history_frequency_grid/`: archived compact top-level summaries from the final history/frequency grid. The winning checkpoint is preserved in `task2_archive_summary/best_checkpoint_from_grid/`.

Important caution: random-split runs in different output directories used different random validation splits. Comparisons across different output directories should be treated as directional. The strongest evidence is the history/frequency grid because its 20 variants shared the same validation split.

## Executive Summary

The current best model is the cleaned C4 current-bits model with a two-layer 24-step z-history GRU:

`task2_best_C4_bits_H24_GRU2_freqRBF16`

The decisive improvement came from history modeling, not from more frequency machinery.

On the shared random split from the history/frequency grid:

| model | best val BCE | delta vs one-layer C4 baseline | count corr | count MAE | best epoch |
|---|---:|---:|---:|---:|---:|
| `hist_H24_D64_layers2` | `0.033972` | `-0.000472` | `0.896922` | `28.249762` | `110` |
| `hist_H40_D64` | `0.034031` | `-0.000413` | `0.893281` | `28.639080` | `116` |
| `hist_H40_D96` | `0.034038` | `-0.000405` | `0.894562` | `28.698530` | `82` |
| one-layer baseline `C4_linear_H24_D64_freq_rbf16` | `0.034444` | `0.000000` | `0.883843` | `29.504976` | `92` |

The final cleaned model corresponds to the winning `hist_H24_D64_layers2` architecture, renamed as the single current-best model.

## What Worked and Is Relevant

### 1. Keeping the C4/current-bits representation

The useful current-pattern representation is the four binary current bits, projected linearly. The current cleaned model has:

- `use_pattern_embedding=False`
- `use_bit_features=True`
- `current_bit_feature_mode="single"`
- `current_bit_encoder="linear"`

Why this worked:

- The 16 stimulation patterns are naturally represented by four bits.
- A direct pattern ID embedding can memorize pattern identity and reduce pressure to learn transferable bit structure.
- The current-bits representation keeps the model compact and makes the pattern representation less arbitrary.

The retained current-bits baseline run achieved:

| model | best val BCE | improvement vs global | improvement vs pattern | count corr | count MAE |
|---|---:|---:|---:|---:|---:|
| `C4_current_bits_linear_trueLPO_random` | `0.034102` | `9.96%` | `5.05%` | `0.872631` | `31.148085` |

That run is not directly comparable to the later grid because it used a different random split, but it confirmed the cleaned C4 current-bits baseline was strong enough to use as the base for the final search.

### 2. Modeling recent stimulation history

The assumption audit already suggested history should matter:

- correlation between previous total response and current total response: `0.3068`
- correlation between previous total response and condition-residual current total response: `0.3719`
- mean current response after high previous-response trials: `138.63`
- mean current response after low previous-response trials: `78.10`

That is a large enough history signal that a purely current-stimulus model leaves information on the table.

The final history grid confirmed this. Short history windows were weak, and the best model used a two-layer GRU over a 24-step history:

| history len | hidden 32 | hidden 64 | hidden 96 |
|---:|---:|---:|---:|
| 8 | `0.034977` | `0.035011` | `0.035011` |
| 16 | `0.034925` | `0.034784` | `0.034744` |
| 24 | `0.034869` | `0.034444` | `0.034297` |
| 40 | `0.034587` | `0.034031` | `0.034038` |

The best single change was not simply making the window longer. It was increasing temporal processing depth:

`H24_D64_layers2`: `0.033972`

Why this worked:

- A short window, especially `8`, misses slower carry-over effects.
- A longer window, especially `40`, helps, but also gives the GRU more sequence to compress.
- A two-layer GRU at length `24` gives a better temporal transform without needing to carry the full `40`-step window.
- The gains were concentrated in the early response time bins, which are the hardest and highest-BCE part of the response.

The best model improved early bins roughly around time bins `6-15`, with the largest per-time-bin improvements versus the one-layer baseline around `0.0014` to `0.0024` BCE.

### 3. Keeping categorical history-pattern embeddings

The history GRU has a separate representation choice from the current-pattern representation. The current pattern works best as bits, but the previous-pattern sequence worked better as categorical embeddings.

In the final grid:

| variant | best val BCE | delta vs baseline |
|---|---:|---:|
| `hist_H24_D64_layers2` | `0.033972` | `-0.000472` |
| `hist_H24_D64_bits` | `0.034694` | `+0.000250` |
| one-layer baseline | `0.034444` | `0.000000` |

Why this happened:

- Current-pattern bits help the model generalize the current stimulus identity.
- History is different: the previous pattern sequence is a temporal state signal, not just a current stimulus descriptor.
- Compressing previous patterns to raw bits appears too restrictive for the history encoder.
- A learned history-pattern embedding lets the GRU learn pattern-specific carry-over states.

So the final model uses:

- current pattern: four raw bits through a linear projection
- history patterns: categorical embeddings inside the GRU

This split is important.

### 4. Causal z-window summary context

The current model keeps causal schedule/context features derived only from previous trials. The active context mode is:

`deploy_context_mode="z_window_summary"`

The context vector has 28 features:

- previous frequency
- time since previous stimulation
- rolling previous frequency over three steps
- current minus previous frequency
- for windows `3`, `5`, `10`, and `20`:
  - previous frequency mean
  - previous frequency standard deviation
  - last frequency
  - current minus last frequency
  - pattern diversity
  - fraction of previous patterns equal to the current pattern

Why this worked:

- It gives the model cheap, causal summaries of the recent stimulation schedule.
- It partly explains why extra direct frequency machinery did not help much: frequency information already appears in the context and in the GRU history continuous inputs.
- The direct-only frequency model without frequency-rich z-window context was worse than the baseline.

### 5. Physical electrode coordinates and graph processing

The current model uses extracted physical electrode coordinates, electrode embeddings, electrode-rate features, and graph message passing.

The retained model config has:

- `use_coordinates=True`
- `use_electrode_embedding=True`
- `use_electrode_rate=True`
- `use_graph=True`
- `n_graph_layers=2`
- `graph_k=8`

Why this is relevant:

- The response tensor is spatial: nearby electrodes tend to have related behavior.
- The graph branch lets each electrode representation use a local neighborhood aggregate before decoding time responses.
- The model does not treat electrodes as exchangeable or independent.
- Earlier cleanup fixed coordinate extraction so the graph uses real physical electrode positions rather than silently broken or arbitrary coordinates.

### 6. Residual decoding around a simple baseline

The model predicts a residual added to fixed baseline logits. In the current best config:

`baseline_mode="global"`

That means the stored baseline logits are a smoothed global response template repeated across patterns, and the neural network learns the stimulus-, history-, frequency-, electrode-, and time-dependent residual.

Why this worked:

- A baseline stabilizes training because the model does not need to learn the whole firing probability surface from scratch.
- A residual model can focus capacity on deviations from a simple average response.
- The global baseline avoids baking a strong fixed pattern lookup table into the output; pattern effects have to come through the learned bit/history/context conditioning.

### 7. Small auxiliary count and PSTH losses

The current config keeps:

- `aux_count_weight=0.03`
- `aux_psth_weight=0.03`

Why this is relevant:

- The main objective remains BCE.
- The auxiliary losses provide weak pressure for total spike count and average temporal profile to stay calibrated.
- The best current model improved both BCE and count metrics compared with the one-layer baseline: count correlation improved from `0.883843` to `0.896922`, and count MAE improved from `29.50` to `28.25`.

## What Did Not Work or Is Not Relevant

### 1. More frequency machinery

The frequency experiments were almost flat or worse:

| variant | best val BCE | delta vs one-layer baseline |
|---|---:|---:|
| `freq_rbf8` | `0.034417` | `-0.000026` |
| `freq_context_only` | `0.034421` | `-0.000023` |
| `freq_film_rbf16` | `0.034439` | `-0.000005` |
| baseline RBF16 | `0.034444` | `0.000000` |
| `freq_fourier_only` | `0.034479` | `+0.000035` |
| `freq_rbf32` | `0.034516` | `+0.000072` |
| `freq_direct_only_nofreq_context` | `0.034545` | `+0.000102` |

Why this did not matter much:

- The schedule context already contains frequency-derived features.
- The z-history continuous inputs also include normalized previous frequencies.
- Removing direct frequency features but keeping context was basically tied with baseline.
- Keeping direct frequency but removing frequency-rich context was worse.
- Increasing RBF resolution from 16 to 32 added capacity but did not improve BCE.
- FiLM modulation added complexity but produced essentially no gain.

Conclusion: frequency is relevant, but the useful frequency information is already captured by simple direct features plus causal context. More elaborate conditioning is not currently worth it.

### 2. FiLM/gated frequency conditioning

A frequency FiLM model was tested:

`freq_film_rbf16`: `0.034439`, only `-0.000005` better than the one-layer baseline.

Why this is not relevant now:

- The improvement is too small to distinguish from run noise.
- It adds parameters and conceptual complexity.
- It did not beat the history improvements.
- The current bottleneck is temporal/history representation, not nonlinear frequency modulation.

### 3. Larger frequency RBF banks

`freq_rbf32` was worse than RBF16:

`0.034516` versus baseline `0.034444`.

Why this failed:

- More bins increase frequency-specific flexibility.
- But the data does not reward that flexibility on the random split.
- Exact or overly granular pattern-frequency baselines in the assumption audit were also worse, indicating frequency-specific memorization can hurt.

The assumption audit showed:

- random global baseline BCE: `0.038079`
- random pattern baseline BCE: `0.036122`
- random exact pattern-frequency baseline BCE: `0.042842`, worse than global
- leave-frequency-out pattern-frequency exact baseline fell back to global-like behavior

That is a warning against sparse frequency memorization.

### 4. Using bits inside the history GRU

`hist_H24_D64_bits` was worse than the baseline:

`0.034694`, delta `+0.000250`.

Why this failed:

- Bit decomposition is good for the current stimulus, where the goal is controlled generalization across pattern structure.
- The history GRU needs to model carry-over state from previous patterns.
- Previous pattern identity may have nonlinear temporal effects not captured by four raw bits.
- Learned embeddings give the recurrent state more expressive tokens.

### 5. Very short history windows

The 8-step history variants were the worst in the grid:

| variant | best val BCE |
|---|---:|
| `hist_H8_D32` | `0.034977` |
| `hist_H8_D64` | `0.035011` |
| `hist_H8_D96` | `0.035011` |

Why this failed:

- The audit showed meaningful previous-response dependence.
- The relevant state is not confined to only a handful of immediately previous trials.
- Short windows underfit the history signal and peak early in training.

### 6. Bigger hidden size alone

Increasing hidden size helped somewhat but did not solve the problem alone:

- `H24_D96`: `0.034297`, better than one-layer H24/D64 but worse than two-layer H24/D64.
- `H40_D96`: `0.034038`, almost tied with H40/D64 and still worse than two-layer H24/D64.

Why this is not the main lever:

- The model benefits more from temporal depth than from simply widening the recurrent state.
- Hidden size `96` adds capacity, but the useful structure appears to be hierarchical temporal processing.

### 7. Running longer as the default answer

The best current model peaked at epoch `110` and stopped at `158`. The one-layer baseline peaked around epoch `92`.

Why more epochs is not the next lever:

- Good models already reached their best validation BCE well before or around the maximum epoch budget.
- Final BCE was slightly worse than best BCE, which indicates mild overfitting after the optimum.
- Checkpointing already preserves the best epoch.

More epochs may be useful only if a future model is still improving at the end. It is not the explanation for the current gap.

### 8. Multi-GPU data parallelism for one model

The GPU-utilization experiments showed that using more GPUs for a single small model was not useful. The issue was more likely CPU/data-loading/overhead than raw GPU compute.

Why it is not relevant to model quality:

- The model is small, around `322,770` parameters for the best two-layer version.
- Multi-GPU synchronization overhead can dominate.
- The final cleaned notebook correctly uses one worker and one GPU for the single best model.

For architecture search, separate one-GPU worker processes were useful. For the current single model, one GPU is enough.

## Current Model Architecture in Detail

### High-level structure

The current model is a coordinate-aware graph temporal residual decoder. Its output is a full spike probability logit tensor per trial:

`[electrodes, time_bins]`

The model predicts:

`logits = baseline_logits + residual_scale * learned_residual`

where `residual_scale` is a learned scalar initialized to `0.2`.

The model is not a plain GRU output head. The GRU encodes stimulation history into a condition vector. That condition vector then modulates a spatial graph decoder and temporal basis decoder.

### Inputs

For each trial, the model consumes:

- current pattern ID, but represented as four binary current bits
- current frequency
- previous pattern ID
- causal z-window context features
- previous 24 stimulation tokens for the z-history GRU
- electrode coordinates and electrode IDs
- fixed graph adjacency between electrodes

The target is the binned spike response tensor:

`[n_electrodes, n_time_bins]`

### Current pattern representation

The current pattern is converted to four bits:

`[(pattern >> 0) & 1, ..., (pattern >> 3) & 1]`

Those four bits are passed through one linear layer:

`4 -> bit_emb_dim=16`

There is no current-pattern ID embedding:

`use_pattern_embedding=False`

This is intentional: the model should represent current patterns through their bit structure.

### Frequency encoder

The direct frequency encoder produces:

- linear normalized frequency
- log normalized frequency
- Fourier features on log-frequency for frequencies `[1, 2, 4, 8]`, with sin and cos pairs
- 16 RBF features over log-frequency

So the direct frequency vector has:

`2 + 8 + 16 = 26` dimensions

The model keeps this simple RBF16 encoder because RBF8 was only barely better in one random split and RBF32 was worse. RBF16 is a stable middle choice.

### Causal z-window context

The model builds a standardized 28-dimensional context vector from previous stimulation schedule only.

It includes previous-frequency and delta-time features plus window summaries over previous `3`, `5`, `10`, and `20` trials. For each window it uses frequency mean, frequency standard deviation, last frequency, current-minus-last frequency, pattern diversity, and same-current-pattern fraction.

This context is deployable because it does not use the current neural response.

### Z-history GRU

The best history encoder uses:

- history length: `24`
- previous pattern mode: categorical embedding
- pattern embedding dim inside history: `16`
- continuous history inputs per step: `3`
  - normalized previous frequency
  - normalized inter-stimulation interval
  - normalized age/validity marker
- GRU hidden dim: `64`
- GRU layers: `2`
- output: final hidden state from the top GRU layer, dimension `64`

Each previous stimulation token is transformed by:

`[history_pattern_embedding, history_continuous_features] -> Linear -> GELU -> LayerNorm`

Then the sequence goes into the two-layer GRU.

Why this is the core current improvement:

- It captures trial-to-trial carry-over effects.
- It performs better than both shorter windows and raw-bit history encoding.
- It improves the hard early time bins and the hard high-BCE patterns.

### Condition vector

The condition vector concatenates:

- current bit projection, dimension `16`
- direct frequency encoding, dimension `26`
- previous pattern embedding, dimension `16`
- causal z-window context, dimension `28`
- z-history GRU state, dimension `64`

Total condition input dimension:

`16 + 26 + 16 + 28 + 64 = 150`

This is passed through the condition MLP:

`150 -> 192 -> 96`

with LayerNorm, GELU, and dropout. The resulting condition vector has dimension `96`.

Frequency FiLM is disabled in the current model.

### Electrode representation

For each electrode, the model builds an electrode feature vector from:

- physical 2D coordinates
- normalized electrode index
- normalized electrode rate feature derived from the baseline response
- learned electrode embedding, dimension `24`
- trial condition vector, dimension `96`

This is passed through an electrode MLP into a hidden state of dimension `96`.

### Graph message passing

The model builds a k-nearest-neighbor electrode graph using physical coordinates:

- `graph_k=8`
- `n_graph_layers=2`

Each graph block aggregates neighbor hidden states with the adjacency matrix and updates electrode states conditioned on the trial condition vector:

`[current electrode hidden, neighbor aggregate, condition] -> MLP -> residual update -> LayerNorm`

This lets each electrode prediction use local spatial context.

### Temporal decoder

The temporal decoder uses a learned basis approach.

For each time bin, it combines:

- learned time embedding
- normalized time
- log-time feature
- Gaussian early-response feature centered around bin `7`
- late-bin indicator for `t >= 20`

These features are mapped to `k_modes=80` temporal basis values.

Each electrode hidden state predicts `80` coefficients. The residual response is produced by:

`einsum(electrode_coefficients, temporal_basis)`

The model also adds:

- electrode-specific spatial bias
- condition-dependent time bias

Then it adds the residual to the fixed baseline logits.

### Loss and optimization

The training loss is:

`BCE + 0.03 * count_loss + 0.03 * psth_loss`

where:

- BCE is the main binary spike-response objective.
- count loss aligns total predicted spike count per trial.
- PSTH loss aligns the average temporal response profile.

Training config:

- batch size: `1024`
- learning rate: `8e-4`
- weight decay: `1e-4`
- dropout: `0.12`
- max epochs: `160`
- validation every `2` epochs
- early stopping patience: `24`
- gradient clipping: `1.0`

## Why the Current Model Is the Best Available Version

The current model keeps the pieces that repeatedly helped:

- current bits rather than current pattern ID embedding
- causal z-window context
- z-history GRU
- categorical history-pattern embeddings
- simple direct RBF16 frequency features
- graph/electrode-aware residual temporal decoder
- small count/PSTH auxiliary losses

It removes or avoids the pieces that did not justify their complexity:

- frequency FiLM/gating
- larger frequency RBF banks
- direct-frequency-only variants without context
- raw-bit history encoding
- very short history windows
- running many variants in the cleaned notebook
- multi-GPU data parallelism for one small model

The most important reason is that the data appears to be history-limited, not frequency-architecture-limited. The audit found meaningful previous-response dependence, and the final grid showed the largest gains from the two-layer history GRU. Frequency matters, but simple frequency/context features already capture most of what is useful.

## Remaining Questions

The current version is the best retained model on a random validation split, but a few questions remain if the goal is stronger generalization:

1. Does `H24_D64_layers2` also win under leave-pattern-out and leave-frequency-out splits?
2. Is `H40_D64_layers2` better than `H24_D64_layers2`, or does the longer window overfit/noise out on harder splits?
3. Should the final submission train on all available training data after hyperparameters are fixed?
4. Are the hard patterns, especially high-BCE odd patterns like `1`, `3`, `5`, `7`, `9`, still the main bottleneck under harder splits?
5. Are the worst electrodes dominated by real biological/spatial structure or by noisy channels that should be handled separately?

Recommended next experiment, if more validation is needed:

- keep the current architecture fixed
- compare only `H24_D64_layers2` vs `H40_D64_layers2` vs `H40_D96` under the harder split of interest
- do not reopen frequency FiLM/RBF architecture search unless those harder splits contradict the current evidence

## Final Current-Best Configuration

Use this as the stable Task 2 model unless a harder validation split proves otherwise:

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
