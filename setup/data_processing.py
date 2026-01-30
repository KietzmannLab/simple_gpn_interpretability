import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from setup.token_world import generate_world_seq
from torch.utils.data import Dataset


def load_data(file_path, test_size=0.2, random_state=42):
    df = pd.read_parquet(file_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df


def process_data(world_seq_map):
    all_tokens, all_directions, all_targets = [], [], []

    for world, sequences in world_seq_map.items():
        token_sequence, direction_sequence = sequences
        tokens = torch.tensor(
            np.array([token.one_hot_vector for token in token_sequence]), dtype=torch.float32
        )
        directions = torch.tensor(direction_sequence, dtype=torch.float32)

        # Target: next-token prediction (remove last token, convert to indices, shift left)
        targets = torch.tensor([torch.argmax(one_hot, dim=-1) for one_hot in tokens[1:]], dtype=torch.long)

        # Truncate inputs to match target length
        tokens = tokens[: targets.size(0)]
        directions = directions[: targets.size(0)]

        all_tokens.append(tokens)
        all_directions.append(directions)
        all_targets.append(targets)

    all_tokens = torch.stack(all_tokens)
    all_directions = torch.stack(all_directions)
    all_targets = torch.stack(all_targets)

    return all_tokens, all_directions, all_targets


# Define a custom dataset
class SeqMapDataset(Dataset):
    def __init__(self, seq_config, seed=None):
        self.seq_config = seq_config
        self.seed = seed if seed is not None else seq_config.get("seed", 42)
        self._rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)


    def __len__(self):
        return self.seq_config["batch_per_epoch"] *  self.seq_config["batch_size"]

    def __getitem__(self, idx):

        input_sequences = generate_world_seq(
            self.seq_config["tokens_list"],
            self.seq_config["world_shape"],
            self.seq_config["tokens_amount"],
            self.seq_config["sequence_length"],
            seq_type = self.seq_config["seq_type"]
        )

        tokens, directions, targets = process_data(input_sequences, self.seq_config["model_type"])
        return tokens, directions, targets
