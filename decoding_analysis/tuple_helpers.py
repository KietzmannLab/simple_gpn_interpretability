

import pickle
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from setup.visual_settings import PALETTE


def _get_metric(results, model_name, key):
    d = results[model_name][key]
    mean = np.asarray(d["mean"], dtype=float)
    ci   = np.asarray(d.get("ci", [0]*len(mean)), dtype=float)
    return mean, ci


def load_metric_files(path_dict):
    """
    loads any *.pkl metric file provided in path_dict.
    """

    loaded = {}

    for key, path in path_dict.items():
        if path is None:
            loaded[key] = None
            continue

        try:
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
        except FileNotFoundError:
            print(f"[WARN] File not found for key '{key}': {path}")
            loaded[key] = None
        except Exception as e:
            print(f"[ERROR] Could not load file for key '{key}': {path}")
            print("  →", e)
            loaded[key] = None

    return loaded

def merge_results(loaded):
    """
    Merge any number of result groups (matched, mismatched, etc.)
    into one result dictionary
    """
    merged = {}

    # iterate over all top-level groups except those that are None
    for group_name, group in loaded.items():
        if group is None:
            continue

        # group: dict(model → metrics)
        for model, metrics in group.items():
            if model not in merged:
                merged[model] = {}

            # merge metrics
            for metric_key, metric_values in metrics.items():
                merged[model][metric_key] = metric_values

    return merged

def finalize_metrics(results):
    for model in results:
        for metric, data in results[model].items():
            vals = data["values"]  # per-layer list of dicts

            means = [v.get("acc_mean", np.nan) for v in vals]
            stds  = [v.get("acc_std", np.nan) for v in vals]
            ci    = [1.96 * s / np.sqrt(len(vals)) for s in stds]

            data["mean"] = means
            data["ci"]   = ci


def plot_pos_label_panels(results, model_name="model"):
    # (pos, label) -> (model_key, combined_key)
    panels = [
        ((1, 1), "pos1_label1", "combined_pos1_label1"),
        ((0, 0), "pos0_label0", "combined_pos0_label0"),
        ((1, 0), "pos1_label0", "combined_pos1_label0"),
        ((0, 1), "pos0_label1", "combined_pos0_label1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()

    for ax, ((p, l), model_key, comb_key) in zip(axes, panels):
        model_mean, model_ci = _get_metric(results, model_name, model_key)
        comb_mean,  comb_ci  = _get_metric(results, model_name, comb_key)

        n_layers = len(model_mean)
        x = np.arange(n_layers)

        # colors: match -> green, mismatch -> pink
        is_match = (p == l)
        comb_color  = PALETTE["pale_green"] if is_match else  PALETTE["pale_pink"]
        model_color = PALETTE["green"] if is_match else  PALETTE["pink"] # keep model distinct; only the match/mismatch rule drives the combined color

        w = 0.36
        ax.bar(x - w/2, comb_mean,  width=w, yerr=comb_ci,  capsize=3,
               label="combined", color=comb_color, alpha=0.9)
        ax.bar(x + w/2, model_mean, width=w, yerr=model_ci, capsize=3,
               label="model", color=model_color, alpha=0.9)

        ax.set_title(f"pos{p}, label{l}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
        ax.axhline(0, linewidth=0.8)

        # cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # one legend for the whole figure
    legend_handles = [
    Patch(facecolor=PALETTE["pale_green"], edgecolor="none", label="combined (match)"),
    Patch(facecolor=PALETTE["pale_pink"],  edgecolor="none", label="combined (mismatch)"),
    Patch(facecolor=PALETTE["green"],      edgecolor="none", label="model (match)"),
    Patch(facecolor=PALETTE["pink"],       edgecolor="none", label="model (mismatch)"),
    ]   

    fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    frameon=False
    )
    fig.suptitle("Tuple Decoding vs Expectation")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def setup_tuple(mode="tuple"):
    # 6 positions: center + pentagon vertices
    center = (0,0)
    radius = 3
    positions = [center] + [
        (
            round(center[0] + radius * math.cos(2 * math.pi * i / 5), 4),
            round(center[1] + radius * math.sin(2 * math.pi * i / 5), 4),
        )
        for i in range(5)
    ]
    if mode=="tuple":
        # (label, position) → id and id → (label, position)
        tuple_to_id = {(label, pos): i for i, (label, pos) in enumerate(product(LETTERS, positions))}
        index_to_tuple = {i: (label, pos) for (label, pos), i in tuple_to_id.items()}
        return tuple_to_id, index_to_tuple
    
    if mode=="position":
        pos_id = {pos: idx for idx, pos in enumerate(positions)}
        index_to_id = {i: pos for  pos, i in pos_id.items()}
        print(pos_id)
        return pos_id, index_to_id