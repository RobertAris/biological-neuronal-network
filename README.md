# In Vitro Neural Response Modeling

This repository contains the final research notebooks for modeling stimulation-response recordings from biological neuronal networks. The work focuses on two complementary prediction problems:

1. infer the stimulation pattern that produced an observed neural response;
2. predict the full electrode-by-time spike response from stimulation parameters.

The repository has been reduced to the final experiments, selected lightweight results, and shared utilities needed to inspect or reproduce those two analyses.

## Results

| task | dataset | target | selected metric |
| --- | --- | --- | ---: |
| Task 1 | `N6_DIV40` | 16-class stimulation-pattern ID | `98.5313%` validation macro-F1 |
| Task 2 | `N5_DIV40` | spike-response tensor `(9839, 105, 80)` | `0.3506 ms` blended weighted W1 proxy |

Task 1 uses a hybrid electrode-time and loop-topology classifier. The selected run reached `98.5288%` validation accuracy with `108 / 7341` validation errors; pattern `2` remained the hardest class with `91.2644%` F1.

Task 2 uses a coordinate-aware graph temporal residual decoder with validation-selected blending against pattern and pattern-frequency baselines. The selected blend improved the weighted W1 proxy by `23.3%` over the raw model and `31.8%` over the pattern baseline.

## Repository Layout

```text
Final_Task/
  Final_Task_1.ipynb          # stimulation-pattern classification
  Final_Task_2.ipynb          # spike-response prediction
  model_iteration_summary.md  # detailed architecture, configuration, and metrics
  task1_outputs/              # committed Task 1 validation summaries
  task2_outputs/README.md     # Task 2 artifact policy and selected metrics
  utils/                      # data loading, saving, and plotting helpers
requirements.txt
```

Large model checkpoints (`*.pth`) and generated Task 2 output directories are intentionally not tracked. The selected checkpoint paths and metrics are documented in `Final_Task/model_iteration_summary.md`.

## Data

The notebooks expect the shared project HDF5 files:

- Task 1 training/evaluation: `N6_DIV40.h5`
- Task 1 final inference: `N6_DIV40_test.h5`
- Task 2 training/evaluation: `N5_DIV40.h5`
- Task 2 final inference: `N5_DIV40_test.h5`

Data access is centralized in `Final_Task/utils/data.py`. The `_test` files are used only for final inference because they omit the supervised target for the corresponding task.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open the final notebooks from the repository root or from `Final_Task/`:

```bash
jupyter notebook Final_Task/Final_Task_1.ipynb
jupyter notebook Final_Task/Final_Task_2.ipynb
```

For a fuller record of the selected architectures, splits, checkpoints, and validation diagnostics, see `Final_Task/model_iteration_summary.md`.
