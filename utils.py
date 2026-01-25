
from datetime import datetime
import glob
import os
from pathlib import Path
import pickle
import random
import re
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch

def create_run_folder(run_name: str | None = None, base_dir: str | Path = "outputs") -> Path:
    """
    Create a folder for a new run under base_dir and return the Path.
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(base_dir)
    run_dir = (base_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created run directory: {run_dir}")
    return run_dir

_EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")

def find_latest_checkpoint(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    ckpts = []
    for p in run_dir.glob("epoch_*.pth"):
        m = _EPOCH_RE.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))

    if not ckpts:
        raise FileNotFoundError(f"No epoch_*.pth checkpoints found in {run_dir}")

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_plot(plt, graphs_dir, title):
    import os
    from datetime import datetime

    # Replace spaces and other unsafe filename characters
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")

    os.makedirs(graphs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_path = os.path.join(graphs_dir, f"{timestamp}_{safe_title}.pdf")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to: {plot_path}")

    plt.show()
    return plot_path

def plot_token_world(world, token_colors, title=None, connections=None, save=False, graphs_dir=None ):
    plt.figure(figsize=(8, 8))
    min_pos, max_pos = world.grid
    ax = plt.gca()

    # Plot grid lines
    step = 1.0  # Grid granularity; can be adjusted
    ticks = np.arange(min_pos, max_pos + step, step)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Invert y-axis so (0,0) is top-left
    # ax.invert_yaxis()

    # Plot tokens
    xs, ys = [], []
    for token in world.tokens:
        y, x = token.coordinates
        xs.append(x)
        ys.append(y)
        plt.scatter(x, y, color=token_colors[token], alpha=0.7, s=500, edgecolors="black")
        plt.text(
            x,
            y,
            token.label,
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )

    if connections is not None:
        # Draw connections
        for token1, token2 in connections:
            y1, x1 = token1.coordinates
            y2, x2 = token2.coordinates
            plt.plot([x1, x2], [y1, y2], linestyle="--", color="gray", alpha=0.6)

    # Axis limits with slight padding
    pad = 0.25
    plt.xlim(min_pos - pad, max_pos + pad)
    plt.ylim(max_pos + pad, min_pos - pad)

    plt.title("Token World Grid")
    if save: 
        save_plot(plt, graphs_dir, "world_grid")


def load_world_sequences(dataset_filename="0_start_6_tokens.pkl", n_worlds=500, path=False):
    """
    Load serialized world sequences from a pickle file.

    Parameters
    ----------
    dataset_filename : str
        Either a filename (relative) or full path to the pickle.
    n_worlds : int
        Number of worlds to load (subset).
    path : bool
        If True, interpret `dataset_filename` as a full absolute path.
    Returns
    -------
    dict
        Mapping of {world: (token_seq, directions_seq)}.
    """
    if path:
        world_seq_path = dataset_filename  # full absolute path provided
    else:
        models_dir = os.environ.get("OUTPUTS_DIR", "/share/klab/sthorat/lventura/models")
        world_seq_path = os.path.join(models_dir, f"../sequences/{dataset_filename}")

    sys.path.append("/share/klab/sthorat/lventura/gaze_integration/scripts/setup")

    with open(world_seq_path, "rb") as f:
        sequences = pickle.load(f)

    return dict(list(sequences.items())[:n_worlds])

def load_model(model, device, checkpoints_dir, checkpoint_file=None):
    if not checkpoint_file:
        checkpoint = max(checkpoints_dir, key=os.path.getmtime)
    else:
        checkpoint = os.path.join(checkpoints_dir, checkpoint_file)

    print(f"loading checkpoint {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    return model

