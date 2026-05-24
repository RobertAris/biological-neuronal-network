# Old Task 1 Runs Summary

Generated: 2026-05-21 11:40:08

These runs predate the current final-data setup. They are not directly comparable to the current Task 1 target, which uses shared `N5_DIV40.h5` with `group_data=False` and `test_mode=False`.

The original trained-run directories were removed after this summary was written, so this file is the retained record of those old Task 1 outputs. The `circularity_analysis/` diagnostic folder is kept separately because it is a reusable topology audit, not a trained-run result.

## Metric Summary

| run | model | networks | best epoch | epochs | best val acc % | best val loss | final train acc % | final val acc % | best pattern | worst pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_task_1_accuracy_20260518_164202 | graph_temporal_proto_bit_cnn | [single/unspecified] | 6 | 19 | 43.2942 | 1.50321 | 75.524 | 40.153 | 11 (64.63%) | 8 (10.00%) |
| task1_exact_multinet_loop_random_20260519_172742 | multinetwork_exact_loop_graph_v3 | [5, 6, 7, 8] | 4 | 5 | 36.4138 | 1.68509 | 35.9672 | 35.9189 | 14 (49.29%) | 2 (23.02%) |

## Conclusions From Old Runs

- The best old Task 1 validation accuracy was `43.29%` from `graph_temporal_proto_bit_cnn`; it peaked early and then overfit, with train accuracy rising while validation accuracy drifted down.
- The multi-network DIV21 loop-topology run did not improve the result. It peaked at `36.41%` validation accuracy and showed uneven transfer across networks, especially weaker validation accuracy on networks 7 and 8.
- Pattern IDs are nominal; the old circular-pattern-class assumption should not be used. Loop/topology information is only useful for electrode/response structure.
- Use these results only as historical architecture evidence. Final Task 1 training should use Network 5 / DIV 40 shared final data only.

## Removed Directories

- `final_task_1_accuracy_20260518_164202`
- `task1_exact_multinet_loop_random_20260519_172742`

## Removed Global Checkpoints

- `/home/bnn_10fs26/best_model_final_task_1_multinetwork_circular_graph.pth`
- `/home/bnn_10fs26/best_model_final_task_1_exact_multinetwork_loop_graph.pth`
- `/home/bnn_10fs26/best_model_final_task_1_graph_proto_bits.pth`
- `/home/bnn_10fs26/best_model_final_task_1_accuracy.pth`
- `/home/bnn_10fs26/best_model_final_task_1_graph_temporal.pth`
