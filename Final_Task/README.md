# Final Task Submission

This directory contains the final notebooks, selected model outputs, and presentation material for the two final tasks.

## Entry Points

- `Final_Task_1.ipynb`: Task 1 stimulation-pattern classification for `N6_DIV40`.
- `Final_Task_2.ipynb`: Task 2 spike-response prediction for `N5_DIV40`.
- `Presentation.pptx`: summary slides for the final presentation.
- `model_iteration_summary.md`: concise record of the selected configurations, checkpoints, and validation metrics.

## Selected Outputs

- Task 1: `task1_outputs/task1_N6_DIV40_macro_f1_seed42_20260525_024957/`
- Task 2: `task2_outputs/hybrid_spike_localization_grid/task2_hybrid_gated_topology_N5_DIV40_grid_robust_blend_tuning/grid_robust_blend_next_16_best_blend_seed7/`

## Data Access

The notebooks load data through `utils.data.load_data`, which resolves the shared datasets from the configured central data path. Local `_test` inference outputs are written through `utils.data.save_data`.
