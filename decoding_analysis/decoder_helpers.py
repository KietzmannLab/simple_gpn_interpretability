
from datetime import datetime
from pathlib import Path
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import Parallel, delayed
from scipy.stats import t 
import torch
from tqdm import tqdm
from data_processing import process_data
from decoders import svm_label_decoder, tuple_decoder
from utils import save_plot


def extract_hidden(model, world_seq, device, return_all=True):
    tokens, directions, _ = process_data(world_seq)

    tokens = tokens.to(device)
    directions = directions.to(device)

    # Only add batch dim if it's missing
    # Expect tokens: [T, D] -> [1, T, D]
    # or already batched: [B, T, D]
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    if directions.dim() == 2:
        directions = directions.unsqueeze(0)

    with torch.no_grad():
        _, hidden_states, _ = model(tokens, directions, return_all_activations=return_all)

    return hidden_states


def collect_hidden_states(model, world_seqs, layers, baseline_model=None, device="cuda", batch_size=500,):
    model = model.to(device).eval()
    
    if baseline_model is not None:
        baseline_model = baseline_model.to(device).eval()

    items = list(world_seqs.items())  # [(world, (tok_seq, dir_seq)), ...]

    batch_tokens = []
    batch_hidden = {m: {L: [] for L in layers} for m in ["model", "baseline"]}

    print("start collection")
    with torch.inference_mode():
        for i in tqdm(range(0, len(items), batch_size), desc="Extracting hidden (batched)"):
            chunk = items[i : i + batch_size]
            # build a batched seq_map; process_data() will stack into [B,T,*]
            seq_map = {world: (tok_seq, dir_seq) for world, (tok_seq, dir_seq) in chunk}

            # keep tok sequences aligned with batch dimension
            batch_tokens.extend([tok_seq for _, (tok_seq, _) in chunk])

            # model activations: dict layer -> [B,T,H] (torch on device)
            hid_model = extract_hidden(model, "gru", seq_map, device)

            for layer in layers:
                h = hid_model[layer]
                # normalize possible [B,1,T,H] -> [B,T,H]
                if h.ndim == 4 and h.shape[1] == 1:
                    h = h[:, 0]
                # move chunk to CPU once
                batch_hidden["model"][layer].append(h.detach().cpu())

            if baseline_model is not None:
                hid_base = extract_hidden(baseline_model, "gru", seq_map, device)
                for layer in layers:
                    h = hid_base[layer]
                    if h.ndim == 4 and h.shape[1] == 1:
                        h = h[:, 0]
                    batch_hidden["baseline"][layer].append(h.detach().cpu())

    # concat chunks -> [N,T,H] numpy
    for layer in layers:
        batch_hidden["model"][layer] = torch.cat(batch_hidden["model"][layer], dim=0).numpy()
        if baseline_model is not None:
            batch_hidden["baseline"][layer] = torch.cat(batch_hidden["baseline"][layer], dim=0).numpy()
        else:
            batch_hidden["baseline"][layer] = None

    print("states collected")
    return batch_tokens, batch_hidden

def save_results(results, tag="label_decoder", results_dir = RESULTS_DIR ):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_path = results_dir / f"{tag}_{timestamp}.pkl"

    with out_path.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to {out_path}")
    return out_path

def load_results(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)



def run_tuple_decoder(
    models, layers, offsets, batch_hidden, world_seqs,
    tuples_dict=None, cv_folds=5, mode="tuple", pos_id=None, min_t=30):
    metrics = [f"pos{pos}_label{label}" for pos, label in offsets]

    results = {
        m: {metric: {"values": [{} for _ in layers]} for metric in metrics}
        for m in models
    }

    jobs = []
    for model_name in models:
        for li, layer in enumerate(layers):
            for coord_off, label_off in offsets:
                metric_key = f"pos{coord_off}_label{label_off}"
                # removed hidden_states here
                jobs.append((model_name, li, coord_off, label_off, metric_key, min_t))

    print(f"[INFO] Total jobs: {len(jobs)}")

    def worker(model_name, li, coord_off, label_off, metric_key, min_t):
        # retrieve hidden states without pickling / copying
        hidden_states = batch_hidden[model_name][li]

        res = tuple_decoder(
            hidden_states=hidden_states,
            world_seqs=world_seqs,
            tuples_dict=tuples_dict,
            coord_offset=coord_off,
            label_offset=label_off,
            cv_folds=cv_folds,
            mode=mode,
            pos_id=pos_id,
            min_t=min_t
        )
        return (model_name, li, metric_key, res)

    n_workers = -1

    results_list = []
    
    with tqdm(total=len(jobs), desc="Tuple decoder", ncols=100) as pbar:
        for out in Parallel(n_jobs=n_workers, batch_size=1)(
            delayed(worker)(*job) for job in jobs
        ):
            results_list.append(out)
            pbar.update(1)

    for model_name, li, metric_key, result in results_list:
        results[model_name][metric_key]["values"][li] = result

    save_results(results, f"{mode}_decoder")
    return results



def run_label_decoder(models, layers, offsets, batch_hidden, world_seqs, cv_folds=5, min_t=30):
    
    # metrics for label decoding
    metrics = [f"label_{off}" for off in offsets]

    results = {
        m: {metric: {"values": [{} for _ in layers]} for metric in metrics}
        for m in models
    }

    # build jobs: (model, layer, target_offset)
    jobs = []
    for model_name in models:
        for li, layer in enumerate(layers):
            for target_offset in offsets:
                metric_key = f"label_{target_offset}"
                jobs.append((model_name, li, target_offset, metric_key, min_t))

    print(f"[INFO] Total jobs: {len(jobs)}")

    def worker(model_name, li, target_offset, metric_key, min_t):
        hidden_states = batch_hidden[model_name][li]
        token_seq_list = [world_seqs[w][0] for w in world_seqs]
        res = svm_label_decoder(
            hidden_states=hidden_states,
            token_seq_list=token_seq_list,
            cv_folds=cv_folds,
            target_offset=target_offset,
            min_t = min_t, 
        )
        return (model_name, li, metric_key, res)

    n_workers = -1

    results_list = []
    
    with tqdm(total=len(jobs), desc="Label decoder", ncols=100) as pbar:
        for out in Parallel(n_jobs=n_workers, batch_size=1)(
            delayed(worker)(*job) for job in jobs
        ):
            results_list.append(out)
            pbar.update(1)

    for model_name, li, metric_key, result in results_list:
        results[model_name][metric_key]["values"][li] = result

    save_results(results, tag=f"label_decoder_{min_t}")
    return results


def extract_label_offset(metric):
    m = re.search(r"label[_+]?(-?\d+)", metric)
    print (metric)
    return int(m.group(1)) if m else None


def plot_decoder(
    results,  layers, metrics_to_plot, ylabels, colors, linestyles,
    graphs_dir=None, filename="decoder_comparison.png", title="Decoding Performance", show_baseline=False, 
    ):


    fig, ax_model = plt.subplots(1, 1, figsize=(5, 5))
    axes_dict = {"model": ax_model}
    handles = {model: [] for model in axes_dict}
    labels  = {model: [] for model in axes_dict}

    for metric in metrics_to_plot:
        for model in axes_dict:
            ax = axes_dict[model]
            vals = results[model][metric]["values"]
            means = np.array([v["acc_mean"] for v in vals])
            errs = []
            for v in vals:
                scores = v["acc_scores"]
                n = len(scores)
                sem = scores.std(ddof=1) / np.sqrt(n)
                ci = t.ppf(0.975, df=n-1) * sem
                errs.append(ci)

            errs = np.array(errs)


            for layer, (m, e) in enumerate(zip(means, errs)):
                c = colors[metric][model]
                ls = linestyles[metric][model]

                # 1) confidence interval: high alpha, no line
                ax.errorbar(
                    layers,
                    means,
                    yerr=errs,
                    fmt="none",          # <- no line, no markers
                    ecolor=c,
                    elinewidth=2.5,
                    capsize=9,
                    capthick=2,
                    alpha=1,
                )

                # 2) line: low alpha, no errorbars
                ax.plot(
                    layers,
                    means,
                    color=c,
                    linestyle=ls,
                    linewidth=3.5,
                    alpha=0.2,
                )

                # 3) markers: opaque (optional but recommended)
                ax.plot(
                    layers,
                    means,
                    linestyle="None",
                    marker="o",
                    markersize=7,
                    markeredgewidth=1.5,
                    markeredgecolor=c,
                    markerfacecolor=c,
                    alpha=1.0,
                )

            # Add one entry per line type and color
            l = extract_label_offset(metric)
            
            handle = plt.Line2D(
                [0], [0],
                linestyle = linestyles[metric][model],
                color= colors[metric][model],
                linewidth=5.0,
            )

            is_baseline = metric.startswith("combined")  
            col = 0 if is_baseline else 1


            handles[model].append(handle)
            labels[model].append(f"{ylabels[metric]}")


    for model, ax in axes_dict.items():
        ax.set_title(title, fontsize=25)
        ax.set_xlabel("Model Layer", fontsize=25)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(layers)
        ax.tick_params(axis="both", labelsize=25)


    ax_model.set_ylabel("Accuracy", fontsize=25)

    for model in axes_dict:
        pairs = list(zip(labels[model], handles[model]))
        pairs.sort(key=lambda x: x[0])  # alphabetical by label
        labels[model], handles[model] = zip(*pairs)


    ax_model.legend(
        handles["model"],
        labels["model"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
        fontsize=25,
    )
    plt.show()
    save_plot(plt, graphs_dir, filename)