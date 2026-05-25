# Task 1 Shipping Model Summary

Updated: 2026-05-24

## Selected Model

- Notebook: `Final_Task_1.ipynb`
- Checkpoint: `/home/bnn_10fs26/best_model_final_task_1_N5_DIV40_macro_f1.pth`
- Model type: `task1_final_N5_DIV40_macro_f1_v1`
- Training data: shared `N5_DIV40.h5`
- Active network: `5`
- DIV: `40`
- Validation split: stratified random by network/frequency/pattern
- Primary metric: macro-F1
- Secondary metric: exact-label accuracy
- Labels: nominal 16-class stimulation pattern IDs

## Best Completed Run

Run folder:

`task1_N5_DIV40_macro_f1_random_20260524_194723/`

| metric | value |
| --- | ---: |
| status | `early_stopped` |
| completed epochs | `38` |
| best epoch | `29` |
| validation macro-F1 | `99.9613%` |
| validation accuracy | `99.9609%` |
| validation weighted-F1 | `99.9609%` |
| validation micro-F1 | `99.9609%` |
| validation loss at best | `0.001491` |
| neural validation errors | `3 / 7674` |

Remaining validation mistakes:

| frequency | true pattern | predicted pattern |
| ---: | ---: | ---: |
| 3 Hz | 5 | 1 |
| 12 Hz | 1 | 5 |
| 38 Hz | 7 | 3 |

## Shipping Notebook Defaults

```python
NETWORK_CANDIDATES = [5]
DIV = 40
group_data = False
TRAIN_TEST_MODE = False
VAL_FRAC = 0.10
RANDOM_SEED = 42
batch_size = 256
USE_FREQUENCY_FEATURES = False
RAW_TIME_WINDOWS = 8
CLASS_LABEL_SMOOTHING = 0.0
```

Pattern IDs are treated as nominal classes. The electrode loop/circular features in the model describe electrode topology only; they are not a circular target or circular loss over pattern IDs.
