# Task 1 Outputs

Final target: shared dataset `N6_DIV40.h5` (`Network=6`, `DIV=40`, `group_data=False`, `test_mode=False`). The `_test` file is used only for final inference.

The primary Task 1 validation/checkpoint metric is macro-F1; exact-label accuracy is reported as a secondary diagnostic. Pattern IDs are nominal 16-class labels, not circular or ordinal targets.

## Included Results

- `task1_N6_DIV40_macro_f1_seed42_20260525_024957/` - selected N6/DIV40 run. Best validation macro-F1 `98.5313%`, validation accuracy `98.5288%`, best epoch `29`, validation errors `108 / 7341`.

## Notebook Configuration

`Pattern_Classification.ipynb` targets `NETWORK_CANDIDATES = [6]`, `DIV = 40`, `RANDOM_SEED = 42`, and a stratified random validation split.

The final inference cell completed successfully and wrote a Task 1 prediction HDF5 with `stimulation_patterns` shape `(9839,)`.
