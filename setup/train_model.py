import os
import string
from datetime import datetime
import torch
import torch.optim as optim
from data_processing import SeqMapDataset
from model import  GP_model
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from utils import create_run_folder, create_run_folders, find_latest_checkpoint, seed_worker



def train_model(model, epochs, learning_rate, seq_config, run_dir, device, resume=False, seed=42):

    print("Starting training...")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []


    dataset = SeqMapDataset(seq_config, seed=seed)

    #data_loader len is set to seg_config["batch_per_epoch"] -> 200 batches of 512 sequences per epoch
    g = torch.Generator()
    g.manual_seed(seed)

    num_workers = seq_config.get("num_workers", 0)

    data_loader = DataLoader(
        dataset,
        batch_size=seq_config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        generator=g,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Resume training if a checkpoint is provided
    if resume:
        latest_checkpoint = find_latest_checkpoint(run_dir)
        print(f"Loading checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed from epoch {start_epoch}")

    else:
        start_epoch = 0

        initial_checkpoint_path = os.path.join(run_dir, f"initial.pth")
        initial_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 0,
            "loss": None,  # No loss yet
        }
        torch.save(initial_checkpoint, initial_checkpoint_path)
        print(f"Initial checkpoint saved as {initial_checkpoint_path}")

        # Log the initial model to WandB
        initial_artifact = wandb.Artifact("checkpoint_initial", type="model")
        initial_artifact.add_file(initial_checkpoint_path)
        wandb.log_artifact(initial_artifact)

    for epoch in range(start_epoch, epochs):

        model.train()
        total_correct = 0
        total_samples = 0
        epoch_losses = []

        with tqdm(total=200, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (tokens, directions, targets) in enumerate(data_loader):

                # move data to device
                tokens = tokens.to(device)
                directions = directions.to(device)
                targets = targets.to(device)

                # Dataset has dim [seq_amount, sequence_length, feature_size] -> seq_amount is currently 1
                # data_loader adds batch_dim
                tokens = tokens.squeeze(1)  # Shape becomes [batch_dim, sequence_length, feature_size]
                directions = directions.squeeze(1)  # Shape becomes [batch_dim, sequence_length, feature_size]
                targets = targets.squeeze(1)  # Shape becomes [batch_dim, sequence_length, feature_size]

                # forward pass
                optimizer.zero_grad()
                logits = model(tokens, directions, return_all_activations=False)  # logits shape before: torch.Size([250, 186, 26])

                # reshape tensors for CrossEntropy
                logits = logits.reshape(-1, logits.size(-1))  # logits shape: torch.Size([46500, 26])
                targets = targets.view(-1)

                # Compute Accuracy
                preds = torch.argmax(logits, dim=-1)  # Predicted classes
                correct = preds.eq(targets).sum().item()  # Correct predictions
                accuracy = correct / targets.numel()  # Accuracy for this batch

                # Accumulate correct predictions and sample count
                total_correct += correct
                total_samples += targets.numel()

                # compute loss
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                wandb.log({"batch_loss": loss.item(), "batch_accuracy": accuracy})

                # Append loss for epoch aggregation
                epoch_losses.append(loss.item())

                # progress bar update
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # Compute aggregated loss/accuracy
        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Log the aggregated metrics with WandB:
        wandb.log({"epoch": epoch, "epoch_accuracy": epoch_accuracy, "epoch_loss": epoch_loss})

        # save checkpoints
        if epoch % 10 == 0 or epoch == epochs - 1:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": epoch_loss,
            }
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch}.pth")

            torch.save(checkpoint, checkpoint_path)
            print(f"saved as {checkpoint_path}")

            artifact = wandb.Artifact(f"checkpoint_epoch_{epoch}", type="model")
            artifact.add_file(checkpoint_path)  # Attach the file
            wandb.log_artifact(artifact)  # Log it with WandB

    return epoch_losses

if __name__ == "__main__":

    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    set_seed(seed)

    model_config = {
        "epochs": 80,
        "learning_rate": 0.0002,
        "n_layers": 3,
        "hidden_size": 512,
        "embedding_size": 200,
        "dropout": 0.00,
        "tokens_size": 26,  # one-hot size for tokens
        "directions_size": 2,  # coordinate dimensions
        "output_size": 26,  # number of token classes
        "layer_norm": False,
        "seed":seed,
    }

    model = GP_model(
        tokens_size=model_config["tokens_size"],
        directions_size=model_config["directions_size"],
        embedding_size=model_config["embedding_size"],
        hidden_size=model_config["hidden_size"],
        dropout=model_config["dropout"],
        n_layers=model_config["n_layers"],
        output_size=model_config["output_size"],
        layer_norm=model_config["layer_norm"],
    )

    seq_config = {
        "tokens_list": list(string.ascii_lowercase),  # Tokens
        "world_shape": (-4, 4),  # World grid dimensions
        "tokens_amount": (4, 6),  # Number of tokens per world
        "sequence_length": 100,  # Length of each sequence
        "batch_size": 512,      # Number of sequences per batch
        "batch_per_epoch": 200,  # Number of batches dynamically generated per epoch
        "seq_type": "gaze",  # Type of sequence: hidden, gaze, fixed_k, transition
    }

    time = datetime.now().strftime("%d%m_%H%M")
    dir_name = (f"{model_config['n_layers']}_{model_config['hidden_size']}_{time}")
    run_dir = create_run_folder("token_prediction", dir_name)

    # Initialize WandB
    wandb.init(
        project="token_prediction",
        name=dir_name,
        config={
            "model_config": model_config,
            "seq_config": seq_config,
        },
    )

    wandb.watch(model, log="all")
    
    train_model(
        model=model, epochs=model_config["epochs"], 
        learning_rate=model_config["learning_rate"], 
        seq_config=seq_config,device=device, 
        run_dir=run_dir, resume=False, 
        seed=seed
    )

    wandb.finish()