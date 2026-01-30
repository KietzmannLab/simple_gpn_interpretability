from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from accuracy_analysis import accuracy_token
from setup.data_processing import process_data
from setup.utils import save_plot
from setup.visual_settings import PALETTE
import seaborn as sns

def collect_logits_error(model,world_seq, min_step=30, max_step=None):
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    tokens_tensor, directions_tensor, targets_tensor = process_data(world_seq)
    error_report = defaultdict(list)
    world = next(iter(world_seq.keys()))

    device = next(model.parameters()).device  # cuda:0 or cpu

    tokens_tensor      = tokens_tensor.to(device)
    directions_tensor  = directions_tensor.to(device)

    model = model.to(device)
    model.eval()

    if tokens_tensor.dim() == 2:
        tokens_tensor = tokens_tensor.unsqueeze(0)        # → [1, sequende_size, 26]
    if directions_tensor.dim() == 2:
        directions_tensor = directions_tensor.unsqueeze(0)  # → [1, sequende_size, 2]

    with torch.no_grad():
        logits = model(tokens_tensor, directions_tensor, return_all_activations=False)
        # print(f"logits: {logits.shape}")
    
    logits = logits[0] 
    targets = targets_tensor.detach().cpu().numpy()
    if targets.ndim == 2 and targets.shape[0] == 1:
        targets = targets[0]
    pred_classes = torch.argmax(logits, dim=-1).cpu().numpy()

    for step in range(min_step, max_step):
        true = targets[step]
        pred = pred_classes[step]
        if pred != true:
            true_label = alphabet[true]
            pred_label = alphabet[pred]
            # if true_label == secret_label:
            #     print( "step: ", step, "hidden label: ",true_label, "pred label ", pred_label)

            error_report[true_label].append((step, pred_label))
            
    return dict(error_report)


def analyze_hidden_errors(model, world_seqs, analysis_window=(30, 60)):
    """
    Collects raw counts of errors 
    Returns a dictionary with raw counts for flexible plotting later.
    """
    error_reports = [collect_logits_error(model, "gru", {world: seqs},  analysis_window[0],  analysis_window[1]) for world, seqs in world_seqs.items()]
    hidden_errors_count = 0
    other_errors = 0
    hidden_in_others = 0
    total_errors=0
    total_steps = 0

    for error_report, (world, seqs) in zip(error_reports, world_seqs.items()):
        # total errors in this sequence
        total_errors += sum(len(entries) for entries in error_report.values())
        hidden_token = next((tok for tok in world.tokens if tok.hidden), None)        
        total_steps += analysis_window[1]-analysis_window[0]
        hidden_errors = error_report.get(hidden_token.label, [])
        hidden_errors_count += len(hidden_errors)

        for label, errors in error_report.items():
            if label != hidden_token.label:  # only look in other tokens
                for _, pred_label in errors:
                    if pred_label == hidden_token.label:
                        hidden_in_others += 1
                    other_errors +=1
        
    return {
        "total_steps" : total_steps,
        "total_errors":total_errors,
        "hidden_errors": hidden_errors_count,
        "other_erros:":  other_errors,
        "hidden_in_others": hidden_in_others
    }


def analyze_changed_errors(model, world_seqs, analysis_window=(30, 60)):
    """
    Collects raw counts of errors for changed-token analysis.
    Returns a dictionary with raw counts for plotting later.
    """
    error_reports = [collect_logits_error(model, {world: seqs},  analysis_window[0],  analysis_window[1]) for world, seqs in world_seqs.items()]

    total_errors_all = 0
    changed_errors_all = 0
    old_in_changed_all = 0
    old_label_other_all = 0
    new_label_other_all = 0
    total_steps = 0
    
    for error_report, (world, seqs) in zip(error_reports, world_seqs.items()):
        changed_token = next((tok for tok in world.tokens if tok.old_label is not None), None)
        total_steps += analysis_window[1]-analysis_window[0]
        # total errors in this sequence
        total_errors = sum(len(entries) for entries in error_report.values())

        # mistakes on the changed token
        changed_list = error_report.get(changed_token.label, [])
        changed_errors = len(changed_list)

        # among those, how many predicted the old label
        old_in_changed = sum(1 for _, pred in changed_list if pred == changed_token.old_label)


       # errors anywhere with old_label or new_label
        old_label_errors = 0
        new_label_errors_other = 0
        for label, errors in error_report.items():
            if label != changed_token.label:  # only look in other tokens
                for _, pred_label in errors:
                    if pred_label == changed_token.old_label:
                        old_label_errors += 1
                    elif pred_label == changed_token.label:
                        new_label_errors_other += 1


        total_errors_all += total_errors
        changed_errors_all += changed_errors
        old_in_changed_all += old_in_changed
        old_label_other_all += old_label_errors

    
    return {
        "total_errors": total_errors_all,
        "changed_errors": changed_errors_all,
        "old_in_changed": old_in_changed_all,
        "old_label_other": old_label_other_all,
        "new_label_other" : new_label_errors_other,
        "total_steps": total_steps,
    }


def compute_binned_accuracy(
    model,
    world_seqs,
    property_fn1,
    device,
    property_fn2=None,
    *,
    prefer="p1",   # "p1" or "p2" if both match
):
    """
    Per-timestep accuracy + 95% CI for two groups.

    Groups:
      - group1: property_fn1(token) == True
      - group2:
          - if property_fn2 is provided: property_fn2(token) == True
          - else: not property_fn1(token)

    Returns:
      timesteps, p1_acc, p1_ci, p2_acc, p2_ci, n1, n2
    """
    correct_1 = defaultdict(int)
    total_1   = defaultdict(int)
    correct_2 = defaultdict(int)
    total_2   = defaultdict(int)

    for world, (tokens, directions) in world_seqs.items():
        world_seq = {world: (tokens, directions)}
        acc = accuracy_token(model, device, world_seq)  # assumed length == len(tokens)-1

        for i in range(len(acc)):
            t_next = tokens[i + 1]

            is1 = bool(property_fn1(t_next))
            is2 = bool(property_fn2(t_next)) if property_fn2 is not None else (not is1)

            # If both provided and both match, choose one group
            if property_fn2 is not None and is1 and is2:
                if prefer == "p2":
                    is1, is2 = False, True
                else:
                    is1, is2 = True, False

            if is1:
                total_1[i] += 1
                correct_1[i] += int(acc[i] == 1)
            elif is2:
                total_2[i] += 1
                correct_2[i] += int(acc[i] == 1)
            else:
                pass

    max_t = max(max(total_1.keys(), default=-1), max(total_2.keys(), default=-1))
    timesteps = np.arange(max_t + 1) if max_t >= 0 else np.array([], dtype=int)

    def _props_and_ci(correct, total):
        p = np.array([correct[t] / total[t] if total[t] > 0 else np.nan for t in timesteps], dtype=float)
        ci = np.array([
            1.96 * np.sqrt(p[idx] * (1 - p[idx]) / total[t]) if total[t] > 1 else np.nan
            for idx, t in enumerate(timesteps)
        ], dtype=float)
        n = np.array([total[t] if total[t] > 0 else 0 for t in timesteps], dtype=int)
        return p, ci, n

    p1, ci1, n1 = _props_and_ci(correct_1, total_1)
    p2, ci2, n2 = _props_and_ci(correct_2, total_2)

    return timesteps, p1, ci1, p2, ci2, n1, n2


def plot_binned_accuracy(timesteps, p_hidden, ci_hidden, p_other, ci_other, pre_expo_step=None, modification_step=None, line_label="addition",
                                 label_modified=" K tokens", label_other="Unchanged tokens",
                                 title="Accuracy per timestep", ylabel="Accuracy",
                                 graphs_dir="graphs_dir"):
    
    plt.figure(figsize=(6, 5))
    ax = plt.gca()  

    palette = sns.color_palette("husl", 8)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c1 = palette[2]
    c2 = palette[0]

    # Plot normal tokens
    plt.plot(timesteps, p_other, label=label_other, marker=".", color=c1, linewidth=2.0)
    plt.fill_between(timesteps,
                     p_other - ci_other,
                     p_other + ci_other,
                     alpha=0.2,  color=c1)

    # Plot hidden tokens
    plt.plot(timesteps, p_hidden, label=label_modified, marker=".",color=c2, linewidth=2.0)
    plt.fill_between(timesteps,
                     p_hidden - ci_hidden,
                     p_hidden + ci_hidden,
                     alpha=0.2,color=c2)
    

    if pre_expo_step:
        # --- Shade pre-exposure ---
        plt.axvspan(
            pre_expo_step,
            modification_step,
            alpha=0.15,
            color=PALETTE[2]
        )

        x_mid = 0.5 * (pre_expo_step + modification_step)
        

        ax.text(
            x_mid,
            0.08,                      
            line_label,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),  # x in data coords, y in axes coords
            color=PALETTE[2],
            alpha=0.9,
            fontweight="bold", 
            fontsize="14"
        )
    else:
        # --- Reintroduction marker ---
        if modification_step:
            plt.axvline(
                modification_step,
                linestyle="--",
                linewidth=1.5,
                zorder=0,      
                alpha=0.30,
                label=line_label,
                color="black"
            )

    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02), # small inset from corner
        frameon=True,
        fontsize=18,
    )
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)

    save_plot(plt, graphs_dir, title)


def plot_changed_errors_bar(
    counts,
    graphs_dir,
    title="Changed Label Error Analysis",
    analysis_window=None,
):
    # --- Unpack ---
    total_errors = counts["total_errors"]
    changed_errors = counts["changed_errors"]
    other_errors = total_errors - changed_errors

    old_in_changed = counts["old_in_changed"]
    old_in_other = counts["old_label_other"]

    other_in_changed = changed_errors - old_in_changed
    other_in_other = other_errors - old_in_other
    if analysis_window is None:
        total_steps = counts.get("total_steps", None)
    else:
        total_steps = analysis_window[1]-analysis_window[0]
    proportion = total_errors / total_steps if total_steps else 0

    if analysis_window:
        title = f"Error Origin in Timesteps {analysis_window[0]}–{analysis_window[1]}"

    # --- Data ---
    categories = ["Changed tokens", "Unchanged tokens"]
    old_label = [
        old_in_changed / total_steps,
        old_in_other / total_steps]

    other_label = [
        other_in_changed / total_steps,
        other_in_other / total_steps]

    x = np.arange(len(categories))
    width = 0.35

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # --- Bars ---
    ax.bar(
        x - width / 2,
        old_label,
        width,
        color=[PALETTE["pink"], PALETTE["green"]],
        label="Old label",
    )

    ax.bar(
        x + width / 2,
        other_label,
        width,
        color=[PALETTE["pale_pink"], PALETTE["pale_green"]],
        label="Other labels",
    )


    # --- Axes & labels ---
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel("Error (%)", fontsize=16)
    ax.set_ylim(0,25)
    ax.set_title(title, fontsize=26, pad=12)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Legend ---
    legend_patches = [
        Patch(facecolor=PALETTE["pink"], label="Old label"),
        Patch(facecolor=PALETTE["pale_pink"], label="Other labels"), 

        Patch(facecolor=PALETTE["green"], label="Old label"),
        Patch(facecolor=PALETTE["pale_green"], label="Other labels"), 
    ]

    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=12,
        frameon=False,
    )


    plt.tight_layout()
    save_plot(plt, graphs_dir, title.replace(" ", "_"))