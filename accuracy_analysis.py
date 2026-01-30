from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
from setup.data_processing import process_data
from setup.utils import save_plot


def accuracy_token(model, device,  world_seq):
    # Load and prepare data
    tokens, directions, targets = process_data(world_seq)

    # Move data to device
    tokens = tokens.to(device)
    directions = directions.to(device)
    targets = targets.to(device)
    model = model.to(device)

    with torch.no_grad():
        rnn_out = model(tokens, directions)
        rnn_out = rnn_out.reshape(-1, rnn_out.size(-1))
        targets = targets.reshape(-1)
        predictions = torch.argmax(rnn_out, dim=-1)
        per_token_accuracy = predictions.eq(targets).float()


    return per_token_accuracy.cpu().numpy()

def average_metrics(measurements):
    """
    averages over a group of sequences the accuracy or loss per token.
    Returns dicts: means, stds, ns per position.
    """
    grouped = {}
    for seq_measures in measurements:
        for pos, value in enumerate(seq_measures):
            if pos not in grouped:
                grouped[pos] = []
            grouped[pos].append(value)

    mean_losses = {}
    std_losses  = {}
    ns          = {}
    
    for pos, values in grouped.items():
        arr = np.array(values)
        mean_losses[pos] = arr.mean()
        std_losses[pos]  = arr.std(ddof=1)
        ns[pos]          = len(arr)

    return mean_losses, std_losses, ns

def plot_distribution(means, ci_values, graphs_dir, title="Metric Distribution",
                    xlabel="Token Position", ylabel="Metric Value", metric="loss"):
    
    positions = sorted(means.keys())
    means = [means[pos] for pos in positions]

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", 8)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c2 = palette[0]

    plt.plot(positions, means,linewidth=3.5,  label="Mean", color=c2)

    # if metric.lower != "accuracy":
    error_interval = [ci_values[pos] for pos in positions]

    # Shade the area between mean-ci and mean+ci
    lower_bound = [m - s for m, s in zip(means, error_interval)]
    upper_bound = [m + s for m, s in zip(means, error_interval)]




    plt.fill_between(positions, lower_bound, upper_bound, alpha=0.2, label="CI 95%", color=c2)


    if metric == "loss":
        plt.ylim(0, 10)
    else: 
         plt.ylim(0, 1)

    plt.title(title,  pad=15) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel,  labelpad=15) 
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    save_name = title.replace(" ", "_")
    save_plot(plt, graphs_dir, f"{save_name}")