import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.data import load_data


# Final-data diagnostic only. Pattern IDs are nominal; class-circle metrics below
# are retained only as an audit that they should not drive Task 1 modeling.
NETWORK_CANDIDATES = [5]
DIV = 40
GROUP_DATA = False
TRAIN_TEST_MODE = False  # supervised training file: shared/N5_DIV40.h5
N_CLASSES = 16
OUT_DIR = Path(__file__).resolve().parent / "task1_outputs" / "circularity_analysis"


def extract_electrode_xy(electrodes, impedance_map, n_electrodes):
    electrodes_arr = np.asarray(electrodes)
    if electrodes_arr.ndim == 2 and electrodes_arr.shape[0] == n_electrodes and electrodes_arr.shape[1] >= 2:
        return electrodes_arr[:, :2].astype(np.float32)
    imp = np.squeeze(np.asarray(impedance_map))
    if imp.ndim != 2:
        return np.stack([np.arange(n_electrodes), np.zeros(n_electrodes)], axis=1).astype(np.float32)
    width = imp.shape[1]
    ids = electrodes_arr.reshape(-1)[:n_electrodes].astype(int)
    return np.stack([ids % width, ids // width], axis=1).astype(np.float32)


def normalize_coords(coords):
    coord_mean = coords.mean(axis=0, keepdims=True)
    coord_std = coords.std(axis=0, keepdims=True) + 1e-6
    return ((coords - coord_mean) / coord_std).astype(np.float32)


def cosine_similarity_matrix(prototypes):
    x = prototypes.reshape(prototypes.shape[0], -1).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x @ x.T


def circular_distances(n_classes):
    idx = np.arange(n_classes)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(d, n_classes - d)


def distance_similarity_summary(sim):
    dist = circular_distances(sim.shape[0])
    rows = {}
    for d in range(1, sim.shape[0] // 2 + 1):
        vals = sim[dist == d]
        rows[d] = float(np.nanmean(vals))
    return rows


def response_center_angles(prototypes, coords_norm):
    center = coords_norm.mean(axis=0)
    electrode_angles = np.arctan2(coords_norm[:, 1] - center[1], coords_norm[:, 0] - center[0])
    angles = []
    concentration = []
    for proto in prototypes:
        activity = proto.mean(axis=1)
        weights = activity - np.percentile(activity, 35)
        weights = np.clip(weights, 0, None)
        if weights.sum() <= 1e-9:
            weights = activity - activity.min()
        if weights.sum() <= 1e-9:
            angles.append(float("nan"))
            concentration.append(float("nan"))
            continue
        z = np.sum(weights * np.exp(1j * electrode_angles)) / (weights.sum() + 1e-9)
        angles.append(float(np.angle(z)))
        concentration.append(float(np.abs(z)))
    return np.asarray(angles), np.asarray(concentration)


def circular_alignment(pattern_angles, response_angles):
    valid = np.isfinite(response_angles)
    if valid.sum() < 4:
        return {"direction": None, "phase_lock": float("nan"), "mean_abs_error_classes": float("nan")}
    p = pattern_angles[valid]
    r = response_angles[valid]
    best = None
    for direction in (1, -1):
        deltas = r - direction * p
        offset = np.angle(np.mean(np.exp(1j * deltas)))
        aligned_error = np.angle(np.exp(1j * (r - direction * p - offset)))
        phase_lock = float(np.abs(np.mean(np.exp(1j * aligned_error))))
        mean_abs_error_classes = float(np.mean(np.abs(aligned_error)) / (2 * math.pi) * N_CLASSES)
        candidate = {
            "direction": "increasing" if direction == 1 else "decreasing",
            "phase_lock": phase_lock,
            "mean_abs_error_classes": mean_abs_error_classes,
            "offset_radians": float(offset),
        }
        if best is None or candidate["phase_lock"] > best["phase_lock"]:
            best = candidate
    return best


def load_records():
    records = []
    for network in NETWORK_CANDIDATES:
        try:
            stimulation_parameters, stimulation_patterns, responses, _, impedance_map, electrodes = load_data(
                network, DIV, GROUP_DATA, test_mode=TRAIN_TEST_MODE
            )
        except Exception as exc:
            print(f"Skipping network {network}: {exc!r}")
            continue
        responses = np.asarray(responses)
        if responses.ndim == 4 and responses.shape[1] == 1:
            responses = responses[:, 0]
        records.append(
            {
                "network": int(network),
                "stimulation_parameters": np.asarray(stimulation_parameters, dtype=np.float32),
                "stimulation_patterns": np.asarray(stimulation_patterns, dtype=np.int64),
                "responses": responses.astype(np.float32),
                "impedance_map": impedance_map,
                "electrodes": np.asarray(electrodes),
            }
        )
    return records


def plot_electrode_positions(records):
    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4), squeeze=False)
    for ax, record in zip(axes[0], records):
        responses = record["responses"]
        coords = extract_electrode_xy(record["electrodes"], record["impedance_map"], responses.shape[1])
        coords_norm = normalize_coords(coords)
        center = coords_norm.mean(axis=0)
        angles = np.arctan2(coords_norm[:, 1] - center[1], coords_norm[:, 0] - center[0])
        order = np.argsort(angles)
        scatter = ax.scatter(coords_norm[:, 0], coords_norm[:, 1], c=angles, cmap="twilight", s=55, edgecolor="black")
        for rank, electrode_idx in enumerate(order):
            x, y = coords_norm[electrode_idx]
            ax.text(x, y, str(electrode_idx), fontsize=6, ha="center", va="center", color="white")
            next_idx = order[(rank + 1) % len(order)]
            nx, ny = coords_norm[next_idx]
            ax.plot([x, nx], [y, ny], color="0.78", linewidth=0.7, zorder=0)
        ax.scatter([center[0]], [center[1]], marker="+", c="red", s=80)
        ax.set_title(f"Network {record['network']} model coordinates")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("normalized x")
        ax.set_ylabel("normalized y")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.75, label="electrode angle")
    fig.suptitle("Task 1 electrode positions used by the current model", y=1.02)
    fig.tight_layout()
    out_path = OUT_DIR / "task1_model_electrode_positions.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_records(records):
    pattern_angles = 2 * math.pi * np.arange(N_CLASSES) / N_CLASSES
    metrics = []
    for record in records:
        y = record["stimulation_patterns"]
        x = np.log1p(record["responses"])
        coords = extract_electrode_xy(record["electrodes"], record["impedance_map"], x.shape[1])
        coords_norm = normalize_coords(coords)

        prototypes = np.zeros((N_CLASSES, x.shape[1], x.shape[2]), dtype=np.float32)
        counts = np.zeros(N_CLASSES, dtype=int)
        for pattern in range(N_CLASSES):
            mask = y == pattern
            counts[pattern] = int(mask.sum())
            if counts[pattern]:
                prototypes[pattern] = x[mask].mean(axis=0)

        sim = cosine_similarity_matrix(prototypes)
        sim_by_dist = distance_similarity_summary(sim)
        response_angles, response_concentration = response_center_angles(prototypes, coords_norm)
        alignment = circular_alignment(pattern_angles, response_angles)
        adjacent = sim_by_dist.get(1, float("nan"))
        opposite = sim_by_dist.get(N_CLASSES // 2, float("nan"))
        near = float(np.nanmean([sim_by_dist[d] for d in (1, 2) if d in sim_by_dist]))
        far = float(np.nanmean([sim_by_dist[d] for d in (6, 7, 8) if d in sim_by_dist]))

        metrics.append(
            {
                "network": record["network"],
                "n_trials": int(len(y)),
                "n_electrodes": int(x.shape[1]),
                "counts_by_pattern": counts.tolist(),
                "similarity_by_circular_distance": sim_by_dist,
                "adjacent_similarity": adjacent,
                "opposite_similarity": opposite,
                "near_similarity_d1_d2": near,
                "far_similarity_d6_d8": far,
                "near_minus_far_similarity": float(near - far),
                "response_center_angles_radians": response_angles.tolist(),
                "mean_response_center_concentration": float(np.nanmean(response_concentration)),
                **alignment,
            }
        )
    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if not records:
        raise RuntimeError("No Task 1 records could be loaded.")
    png_path = plot_electrode_positions(records)
    metrics = analyze_records(records)
    metrics_path = OUT_DIR / "task1_circularity_metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"records": metrics, "electrode_position_png": str(png_path)}, f, indent=2)

    print(f"Loaded networks: {[r['network'] for r in records]}")
    print(f"Saved electrode position PNG: {png_path}")
    print(f"Saved circularity metrics: {metrics_path}")
    for row in metrics:
        print(
            "Network {network}: near-far={near_minus_far_similarity:.4f}, "
            "adjacent={adjacent_similarity:.4f}, opposite={opposite_similarity:.4f}, "
            "phase_lock={phase_lock:.4f}, mean_abs_error_classes={mean_abs_error_classes:.2f}, "
            "direction={direction}".format(**row)
        )


if __name__ == "__main__":
    main()
