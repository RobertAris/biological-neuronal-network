# Task 2 Outputs

Task 2 generated artifacts are intentionally kept out of git because the selected runs include large checkpoints and output tensors.

## Selected Run

`hybrid_spike_localization_grid/task2_hybrid_gated_topology_N5_DIV40_grid_robust_blend_tuning/grid_robust_blend_next_16_best_blend_seed7/`

| metric | value |
| --- | ---: |
| final blended weighted W1 proxy | `0.3506 ms` |
| raw model weighted W1 proxy | `0.4569 ms` |
| pattern baseline weighted W1 proxy | `0.5141 ms` |
| pattern-frequency baseline weighted W1 proxy | `0.4392 ms` |
| blend improvement over raw model | `23.3%` |
| blend improvement over pattern baseline | `31.8%` |
| best epoch | `156` |
| epochs run | `176` |

The final inference cell in `../Final_Task_2.ipynb` writes `binned_spike_train_responses` with shape `(9839, 105, 80)`.
