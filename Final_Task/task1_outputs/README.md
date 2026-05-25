# Task 1 Outputs

Current training target: shared final dataset `N5_DIV40.h5` (`Network=5`, `DIV=40`, `group_data=False`, `test_mode=False`). The primary Task 1 validation/checkpoint metric is macro-F1; accuracy is secondary. The `_test` file is for inference/submission reference only.

## Current Contents

- `task1_N5_DIV40_macro_f1_random_20260524_194723/` - completed final-data seed-42 random split run with the hybrid raw-electrode-time + graph model. Best validation macro-F1 `99.9613%`, validation accuracy `99.9609%`, best epoch `29`. Ridge baseline macro-F1 `97.4518%`, accuracy `97.4590%`.
- `task1_final_runs_summary.md` - retained concise summary of current final-data Task 1 runs and split status.
- `old_task1_runs_summary.md` - retained summary of old non-final Task 1 runs that were removed because they are no longer comparable.
- `circularity_analysis/` - reusable electrode/topology diagnostic plots and metrics used to audit the old loop prior.

## Current Notebook Default

`Final_Task_1.ipynb` now defaults to `RANDOM_SEED = 1337` for a fresh stratified random validation split overfitting check. New run folders include the split seed in the name, for example:

`task1_N5_DIV40_macro_f1_random_seed1337_<timestamp>/`

To test overfitting, retrain from scratch after changing the seed. Do not only evaluate the old seed-42 checkpoint on the seed-1337 split because most seed-1337 validation samples were seed-42 training samples.
