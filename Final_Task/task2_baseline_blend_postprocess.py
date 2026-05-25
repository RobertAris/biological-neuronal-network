import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/bnn_10fs26/pands_ibnnwml/Final_Task")
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_data


NETWORK = 5
DIV = 40
GROUP_DATA = False
RANDOM_SEED = 42
VAL_FRAC = 0.25
N_PATTERNS_MIN = 16
BINS_PER_MS = 4
ALPHA_GRID = np.asarray([i / 20 for i in range(21)], dtype=np.float32)
MIN_TRIALS = 8
ELECTRODE_MIN_TRUE_MASS = 1e-4
EPS = 1e-8

OUT_DIR = (
    PROJECT_ROOT
    / "task2_outputs"
    / "baseline_blend_postprocess"
    / "pattern_vs_pattern_frequency_seed42"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = (
    PROJECT_ROOT
    / "task2_outputs"
    / "metric_aligned_adaptive_graph_psth"
    / "task2_metric_aligned_adaptive_graph_psth_N5_DIV40_shared_final_random"
    / "best_run"
    / "task2_metric_aligned_adaptive_graph_psth"
    / "split_indices.npz"
)


def standardize_response_array(x):
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected response array [trials,electrodes,time], got {x.shape}")
    return x.astype(np.float32)


def extract_freq_pattern(stim_params, stim_patterns=None):
    z = np.asarray(stim_params)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("stimulation_parameters must have at least two columns: [frequency, pattern].")
    freq = z[:, 0].astype(np.float32)
    if stim_patterns is None:
        pattern = z[:, 1].astype(np.int64)
    else:
        pattern = np.asarray(stim_patterns).astype(np.int64)
    return freq, pattern


def np_logit(p, eps=1e-5):
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def compute_baseline_logits(x_train, patterns_train, n_patterns, alpha=25.0, eps=1e-5):
    global_p = x_train.mean(axis=0).astype(np.float32)
    baseline_p = np.zeros((n_patterns, x_train.shape[1], x_train.shape[2]), dtype=np.float32)
    for p in range(n_patterns):
        mask = patterns_train == p
        if int(mask.sum()) == 0:
            baseline_p[p] = global_p
        else:
            n = int(mask.sum())
            baseline_p[p] = (x_train[mask].sum(axis=0) + alpha * global_p) / (n + alpha)
    return np_logit(baseline_p, eps=eps), global_p


def compute_pattern_frequency_baseline_logits(
    x_train,
    patterns_train,
    freqs_train,
    n_patterns,
    freq_values,
    pattern_baseline_logits,
    alpha=5.0,
    activity_alpha_boost=1.0,
    activity_alpha_power=1.0,
    eps=1e-5,
):
    pattern_p = 1.0 / (1.0 + np.exp(-pattern_baseline_logits))
    freq_values = np.asarray(freq_values, dtype=np.float32)
    baseline_p = np.zeros((n_patterns, len(freq_values), x_train.shape[1], x_train.shape[2]), dtype=np.float32)
    patterns_train = np.asarray(patterns_train).astype(int)
    freqs_train = np.asarray(freqs_train).astype(np.float32)

    pattern_activity = pattern_p.mean(axis=(1, 2)).astype(np.float32)
    lo = float(np.nanmin(pattern_activity))
    hi = float(np.nanmax(pattern_activity))
    activity_norm = (pattern_activity - lo) / max(hi - lo, 1e-8)
    activity_shrink = np.power(np.clip(1.0 - activity_norm, 0.0, 1.0), float(activity_alpha_power))
    alpha_by_pattern = float(alpha) * (1.0 + float(activity_alpha_boost) * activity_shrink)

    for p in range(n_patterns):
        prior = pattern_p[p]
        alpha_eff = float(alpha_by_pattern[p])
        for fi, f in enumerate(freq_values):
            mask = (patterns_train == p) & (freqs_train == f)
            n = int(mask.sum())
            if n == 0:
                baseline_p[p, fi] = prior
            else:
                baseline_p[p, fi] = (x_train[mask].sum(axis=0) + alpha_eff * prior) / (n + alpha_eff)
    return np_logit(baseline_p, eps=eps), freq_values, alpha_by_pattern.astype(np.float32)


def pattern_frequency_probs_from_logits(pf_logits, freq_values, pattern_ids, freq_arr):
    freq_values = np.asarray(freq_values, dtype=np.float32)
    pattern_ids = np.asarray(pattern_ids).astype(int)
    freq_arr = np.asarray(freq_arr).astype(np.float32)
    freq_indices = np.abs(freq_arr[:, None] - freq_values[None, :]).argmin(axis=1)
    return 1.0 / (1.0 + np.exp(-pf_logits[pattern_ids, freq_indices]))


def aggregate_1ms_bins(arr, bins_per_ms=BINS_PER_MS):
    arr = np.asarray(arr, dtype=np.float32)
    pad = (-arr.shape[-1]) % int(bins_per_ms)
    if pad:
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)], mode="constant")
    n_ms = arr.shape[-1] // int(bins_per_ms)
    return arr.reshape(*arr.shape[:-1], n_ms, int(bins_per_ms)).sum(axis=-1)


def condition_weighted_w1_ms(true_psth, pred_psth, eps=EPS):
    true_mass = true_psth.sum(axis=-1)
    pred_mass = pred_psth.sum(axis=-1)
    active = true_mass > eps
    if not np.any(active):
        return dict(weighted_sum=0.0, weight_total=0.0, weighted_w1_ms=np.nan, mean_w1_ms=np.nan, n_electrodes=0)
    true_dist = true_psth[active] / (true_mass[active, None] + eps)
    pred_dist = (pred_psth[active] + eps) / (pred_mass[active, None] + eps * pred_psth.shape[-1])
    w1 = np.abs(np.cumsum(pred_dist, axis=-1) - np.cumsum(true_dist, axis=-1)).sum(axis=-1)
    weights = true_mass[active]
    weighted_sum = float((w1 * weights).sum())
    weight_total = float(weights.sum())
    return dict(
        weighted_sum=weighted_sum,
        weight_total=weight_total,
        weighted_w1_ms=float(weighted_sum / max(weight_total, eps)),
        mean_w1_ms=float(np.mean(w1)),
        n_electrodes=int(active.sum()),
    )


def psth_wasserstein_ms(y_true, pred_prob, pats, freqs_arr):
    y_ms = aggregate_1ms_bins(y_true)
    p_ms = aggregate_1ms_bins(pred_prob)
    rows = []
    weighted_sum = 0.0
    weight_total = 0.0
    unweighted = []
    pats = np.asarray(pats).reshape(-1)
    freqs_arr = np.asarray(freqs_arr).reshape(-1)
    for p in np.unique(pats):
        for f in np.unique(freqs_arr[pats == p]):
            mask = (pats == p) & (freqs_arr == f)
            if not np.any(mask):
                continue
            true_psth = y_ms[mask].mean(axis=0)
            pred_psth = p_ms[mask].mean(axis=0)
            score = condition_weighted_w1_ms(true_psth, pred_psth)
            if not np.isfinite(score["weighted_w1_ms"]):
                continue
            weighted_sum += score["weighted_sum"]
            weight_total += score["weight_total"]
            unweighted.append(score["mean_w1_ms"])
            rows.append(dict(pattern=int(p), frequency=float(f), n_trials=int(mask.sum()), **score))
    return dict(
        weighted_w1_ms=float(weighted_sum / max(weight_total, EPS)) if weight_total > 0 else np.nan,
        mean_w1_ms=float(np.nanmean(unweighted)) if unweighted else np.nan,
        n_pairs=int(len(rows)),
        rows=rows,
    )


def blend(a, b, alpha):
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    return alpha_arr * a + (1.0 - alpha_arr) * b


def best_alpha(y_true, pf_prob, pattern_prob, pats, freqs_arr, alpha_grid=ALPHA_GRID):
    records = []
    best = None
    for alpha in alpha_grid:
        pred = blend(pf_prob, pattern_prob, float(alpha))
        score = psth_wasserstein_ms(y_true, pred, pats, freqs_arr)
        rec = dict(alpha=float(alpha), weighted_w1_ms=score["weighted_w1_ms"], mean_w1_ms=score["mean_w1_ms"], n_pairs=score["n_pairs"])
        records.append(rec)
        if best is None or rec["weighted_w1_ms"] < best["weighted_w1_ms"]:
            best = rec
    return best, records


def fit_baseline_blend(y_true, pf_prob, pattern_prob, pats, freqs_arr, electrode_level=False):
    global_best, global_records = best_alpha(y_true, pf_prob, pattern_prob, pats, freqs_arr)
    alpha_by_trial = np.full(len(pats), float(global_best["alpha"]), dtype=np.float32)
    selection_rows = []

    pattern_choice = {}
    for p in np.unique(pats):
        mask = pats == p
        best, _ = best_alpha(y_true[mask], pf_prob[mask], pattern_prob[mask], pats[mask], freqs_arr[mask])
        pattern_choice[int(p)] = float(best["alpha"])

    for p in np.unique(pats):
        for f in np.unique(freqs_arr[pats == p]):
            mask = (pats == p) & (freqs_arr == f)
            n = int(mask.sum())
            if n >= MIN_TRIALS:
                best, _ = best_alpha(y_true[mask], pf_prob[mask], pattern_prob[mask], pats[mask], freqs_arr[mask])
                alpha = float(best["alpha"])
                selected_by = "pattern_frequency"
            else:
                alpha = float(pattern_choice.get(int(p), global_best["alpha"]))
                selected_by = "pattern_fallback"
            alpha_by_trial[mask] = alpha
            selection_rows.append(dict(pattern=int(p), frequency=float(f), n_trials=n, alpha=alpha, selected_by=selected_by))

    pred = blend(pf_prob, pattern_prob, alpha_by_trial[:, None, None])
    electrode_rows = []
    if electrode_level:
        alpha_by_trial_electrode = alpha_by_trial[:, None].repeat(y_true.shape[1], axis=1)
        y_ms = aggregate_1ms_bins(y_true)
        pf_ms = aggregate_1ms_bins(pf_prob)
        pat_ms = aggregate_1ms_bins(pattern_prob)
        for p in np.unique(pats):
            for f in np.unique(freqs_arr[pats == p]):
                mask = (pats == p) & (freqs_arr == f)
                if int(mask.sum()) < MIN_TRIALS:
                    continue
                true_psth = y_ms[mask].mean(axis=0)
                pf_psth = pf_ms[mask].mean(axis=0)
                pat_psth = pat_ms[mask].mean(axis=0)
                true_mass = true_psth.sum(axis=-1)
                for e in np.flatnonzero(true_mass > ELECTRODE_MIN_TRUE_MASS):
                    best_e = None
                    for alpha in ALPHA_GRID:
                        candidate = blend(pf_psth[e:e + 1], pat_psth[e:e + 1], float(alpha))
                        score = condition_weighted_w1_ms(true_psth[e:e + 1], candidate)
                        rec = dict(alpha=float(alpha), w1_ms=float(score["weighted_w1_ms"]))
                        if best_e is None or rec["w1_ms"] < best_e["w1_ms"]:
                            best_e = rec
                    alpha_by_trial_electrode[mask, e] = float(best_e["alpha"])
                    electrode_rows.append(dict(pattern=int(p), frequency=float(f), electrode=int(e), n_trials=int(mask.sum()), true_mass=float(true_mass[e]), **best_e))
        pred = alpha_by_trial_electrode[:, :, None] * pf_prob + (1.0 - alpha_by_trial_electrode[:, :, None]) * pattern_prob

    score = psth_wasserstein_ms(y_true, pred, pats, freqs_arr)
    summary = dict(
        weighted_w1_ms=float(score["weighted_w1_ms"]),
        mean_w1_ms=float(score["mean_w1_ms"]),
        n_pairs=int(score["n_pairs"]),
        global_alpha=float(global_best["alpha"]),
        mean_trial_alpha=float(np.mean(alpha_by_trial)),
        fraction_pf_alpha=float(np.mean(alpha_by_trial >= 0.999)),
        fraction_pattern_alpha=float(np.mean(alpha_by_trial <= 0.001)),
        electrode_level=bool(electrode_level),
        n_electrode_alphas=int(len(electrode_rows)),
    )
    return pred, summary, pd.DataFrame(selection_rows), pd.DataFrame(electrode_rows), pd.DataFrame(global_records)


def nested_eval(y_true, pf_prob, pattern_prob, pats, freqs_arr, seed=20260524, eval_fraction=0.5):
    rng = np.random.default_rng(seed)
    fit_idx = []
    eval_idx = []
    for p in np.unique(pats):
        for f in np.unique(freqs_arr[pats == p]):
            idx = np.flatnonzero((pats == p) & (freqs_arr == f))
            rng.shuffle(idx)
            n_eval = max(1, int(round(len(idx) * eval_fraction)))
            eval_idx.extend(idx[:n_eval].tolist())
            fit_idx.extend(idx[n_eval:].tolist())
    fit_idx = np.asarray(sorted(fit_idx), dtype=int)
    eval_idx = np.asarray(sorted(eval_idx), dtype=int)
    _, fit_summary, fit_rows, _, _ = fit_baseline_blend(
        y_true[fit_idx], pf_prob[fit_idx], pattern_prob[fit_idx], pats[fit_idx], freqs_arr[fit_idx], electrode_level=False
    )
    alpha_lookup = {(int(r.pattern), float(r.frequency)): float(r.alpha) for r in fit_rows.itertuples(index=False)}
    alpha_by_eval = np.full(len(eval_idx), float(fit_summary["global_alpha"]), dtype=np.float32)
    for j, i in enumerate(eval_idx):
        alpha_by_eval[j] = alpha_lookup.get((int(pats[i]), float(freqs_arr[i])), float(fit_summary["global_alpha"]))
    pred_eval = blend(pf_prob[eval_idx], pattern_prob[eval_idx], alpha_by_eval[:, None, None])
    eval_score = psth_wasserstein_ms(y_true[eval_idx], pred_eval, pats[eval_idx], freqs_arr[eval_idx])
    return dict(
        seed=int(seed),
        n_fit=int(len(fit_idx)),
        n_eval=int(len(eval_idx)),
        fit_weighted_w1_ms=float(fit_summary["weighted_w1_ms"]),
        eval_weighted_w1_ms=float(eval_score["weighted_w1_ms"]),
        eval_mean_w1_ms=float(eval_score["mean_w1_ms"]),
    )


def main():
    stimulation_parameters, stimulation_patterns, x, *_ = load_data(NETWORK, DIV, GROUP_DATA, test_mode=False)
    x = standardize_response_array(x)
    freqs, patterns = extract_freq_pattern(stimulation_parameters, stimulation_patterns)
    n_patterns = int(max(N_PATTERNS_MIN, patterns.max() + 1))

    if SPLIT_PATH.exists():
        split = np.load(SPLIT_PATH)
        train_idx = split["train_idx"].astype(int)
        val_idx = split["val_idx"].astype(int)
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.arange(len(x))
        rng.shuffle(idx)
        n_val = int(round(len(idx) * VAL_FRAC))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    pattern_logits, global_p = compute_baseline_logits(x[train_idx], patterns[train_idx], n_patterns, alpha=25.0)
    freq_values = np.sort(np.unique(freqs[train_idx])).astype(np.float32)
    pf_logits, freq_values, alpha_by_pattern = compute_pattern_frequency_baseline_logits(
        x[train_idx],
        patterns[train_idx],
        freqs[train_idx],
        n_patterns,
        freq_values,
        pattern_logits,
        alpha=5.0,
        activity_alpha_boost=1.0,
        activity_alpha_power=1.0,
    )

    pats = patterns[val_idx]
    fvals = freqs[val_idx]
    y_true = x[val_idx]
    pattern_prob = 1.0 / (1.0 + np.exp(-pattern_logits[pats]))
    pf_prob = pattern_frequency_probs_from_logits(pf_logits, freq_values, pats, fvals)

    rows = []
    predictor_scores = {
        "global": psth_wasserstein_ms(y_true, np.broadcast_to(global_p, y_true.shape), pats, fvals),
        "pattern": psth_wasserstein_ms(y_true, pattern_prob, pats, fvals),
        "pattern_frequency": psth_wasserstein_ms(y_true, pf_prob, pats, fvals),
    }
    for name, score in predictor_scores.items():
        rows.append(dict(predictor=name, weighted_w1_ms=score["weighted_w1_ms"], mean_w1_ms=score["mean_w1_ms"], n_pairs=score["n_pairs"]))

    best_global, global_curve = best_alpha(y_true, pf_prob, pattern_prob, pats, fvals)
    global_blend_prob = blend(pf_prob, pattern_prob, best_global["alpha"])
    global_score = psth_wasserstein_ms(y_true, global_blend_prob, pats, fvals)
    rows.append(dict(predictor="baseline_global_alpha_blend", weighted_w1_ms=global_score["weighted_w1_ms"], mean_w1_ms=global_score["mean_w1_ms"], n_pairs=global_score["n_pairs"]))

    cond_prob, cond_summary, cond_rows, _, cond_curve = fit_baseline_blend(y_true, pf_prob, pattern_prob, pats, fvals, electrode_level=False)
    rows.append(dict(predictor="baseline_condition_blend", weighted_w1_ms=cond_summary["weighted_w1_ms"], mean_w1_ms=cond_summary["mean_w1_ms"], n_pairs=cond_summary["n_pairs"]))

    elec_prob, elec_summary, elec_rows, electrode_rows, _ = fit_baseline_blend(y_true, pf_prob, pattern_prob, pats, fvals, electrode_level=True)
    rows.append(dict(predictor="baseline_condition_electrode_blend", weighted_w1_ms=elec_summary["weighted_w1_ms"], mean_w1_ms=elec_summary["mean_w1_ms"], n_pairs=elec_summary["n_pairs"]))

    nested = nested_eval(y_true, pf_prob, pattern_prob, pats, fvals)

    metric_df = pd.DataFrame(rows).sort_values("weighted_w1_ms")
    metric_df.to_csv(OUT_DIR / "baseline_blend_w1_metrics.csv", index=False)
    pd.DataFrame(global_curve).to_csv(OUT_DIR / "global_alpha_curve.csv", index=False)
    cond_rows.to_csv(OUT_DIR / "condition_alpha_choices.csv", index=False)
    electrode_rows.to_csv(OUT_DIR / "condition_electrode_alpha_choices.csv", index=False)
    pd.DataFrame({"pattern": np.arange(n_patterns), "pattern_frequency_alpha": alpha_by_pattern}).to_csv(
        OUT_DIR / "pattern_frequency_alpha_by_pattern.csv", index=False
    )
    summary = dict(
        train_trials=int(len(train_idx)),
        val_trials=int(len(val_idx)),
        split_path=str(SPLIT_PATH),
        min_trials=int(MIN_TRIALS),
        electrode_min_true_mass=float(ELECTRODE_MIN_TRUE_MASS),
        best_predictor=str(metric_df.iloc[0]["predictor"]),
        best_weighted_w1_ms=float(metric_df.iloc[0]["weighted_w1_ms"]),
        pattern_weighted_w1_ms=float(predictor_scores["pattern"]["weighted_w1_ms"]),
        pattern_frequency_weighted_w1_ms=float(predictor_scores["pattern_frequency"]["weighted_w1_ms"]),
        global_alpha=float(best_global["alpha"]),
        condition_blend=cond_summary,
        condition_electrode_blend=elec_summary,
        nested_condition_blend_eval=nested,
        old_grid_best_blended_w1_ms=0.36406076128601894,
        new_metric_aligned_blended_w1_ms=0.36050264456854386,
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(metric_df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
