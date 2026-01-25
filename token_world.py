
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl


class TokenInstance:
    def __init__(self, label, coord, labels_list, hidden=False):
        self.coordinates = coord
        self.label = label
        self.one_hot_vector = self.create_one_hot_vector(labels_list)
        self.hidden = hidden
        self.old_label = None

    def create_one_hot_vector(self, labels_list):
        index = labels_list.index(self.label)
        one_hot = np.zeros(len(labels_list))
        one_hot[index] = 1
        return one_hot

    def set_coordinates(self, x, y):
        self.coordinates = (x, y)



class TokenWorld:
    
    def __init__(self, labels, grid_size, num_tokens, min_distance=0.25, no_repeats=False,  fixed_k=False, put_k=None, rng=None):
       
        """
        labels (list[str]): possible labels
        grid_size (tuple[int,int]): (min, max) world extent
        num_tokens (tuple[int,int]): (min,max) number of tokens

        """
        self.labels = labels
        self.tokens = []
        self.grid = grid_size
        self.min_distance = min_distance
        self.rng = rng or np.random.default_rng()
        self.populate_world(num_tokens=num_tokens, no_repeats=no_repeats,  fixed_k=fixed_k, put_k=put_k)
        

    def populate_world(self, num_tokens, no_repeats=False, fixed_k=False, put_k=None):
        chosen_token_amount = self.rng.integers(num_tokens[0], num_tokens[1] + 1)
        center_coords = (0, 0)
        if put_k is None:
            put_k = random.choice([True, False])
        
        available_labels = set(self.labels)

        if fixed_k:
            available_labels -= {"k"}

        if fixed_k and put_k:
            chosen_token_amount = chosen_token_amount - 1
            k = TokenInstance("k", (1,1), self.labels)
            self.tokens.append(k)


        # Choose center label
        center_label = self.rng.choice(list(available_labels))

        if no_repeats:
            available_labels -= {center_label}

        available_labels = list(available_labels)

        # Place center token
        token = TokenInstance(center_label, center_coords, self.labels)
        token.set_coordinates(0, 0)
        self.tokens.append(token)

        # Place remaining tokens
        for _ in range(chosen_token_amount - 1):
            label = self.rng.choice(available_labels)
            coords = self.get_random_pos(
                min_pos=self.grid[0], max_pos=self.grid[1],
                placed_tokens=self.tokens, rng=self.rng, fixed_k=fixed_k)

            token = TokenInstance(label, coords, self.labels)
            self.tokens.append(token)

            if no_repeats:
                available_labels = list(set(available_labels) - {label})
            else:
                available_labels = self.labels


    def get_random_token(self, rng=None):
        rng = rng or self.rng
        return rng.choice(self.tokens)


    def get_random_pos(self, min_pos, max_pos, placed_tokens,  rng=None, fixed_k=False):
        rng = rng or self.rng

        center = np.array([1.0, 1.0], dtype=float)


        for _ in range(10):
            x = rng.uniform(min_pos, max_pos)
            y = rng.uniform(min_pos, max_pos)
            new_arr = np.array([x, y], dtype=float)

            ok_tokens = all(
                np.linalg.norm(np.array(token.coordinates, dtype=float) - new_arr) >= self.min_distance
                for token in placed_tokens
            )
            ok_center = (not fixed_k) or (np.linalg.norm(new_arr - center) >= self.min_distance)

            if ok_tokens and ok_center:
                return (x, y)

        raise ValueError("Could not place token without overlap.")

    def display_tokens(self):
        for token in self.tokens:
            print(f"Token '{token.label}' at {token.coordinates}")


def assign_token_colors(tokens, colormap="Paired"):
    cmap = plt.colormaps.get_cmap(colormap)
    sorted_tokens = sorted(tokens, key=lambda t: (t.label, id(t)))
    return {t: cmap(i / len(sorted_tokens)) for i, t in enumerate(sorted_tokens)}


def generate_gaze_sequence(world, sequence_length, start_token=None, changed_token=None, rng=None):
    rng = rng or world.rng
    token_sequence, displacement_sequence = [], []
    token_zero = next(t for t in world.tokens if t.coordinates == (0, 0))
    current_token = start_token if start_token else token_zero

    token_sequence.append(current_token)
    for _ in range(sequence_length):
        next_token = world.get_random_token(rng=rng)
        while next_token == current_token:
            next_token = world.get_random_token(rng=rng)
        delta = (next_token.coordinates[0] - current_token.coordinates[0],
                 next_token.coordinates[1] - current_token.coordinates[1])
        displacement_sequence.append(delta)
        token_sequence.append(next_token)
        current_token = next_token
    return token_sequence, displacement_sequence


def generate_hidden_token_seq(world, seq_length, rng=None):
    rng = rng or world.rng
    tokens = world.tokens
    hidden_token = rng.choice(tokens)
    hidden_token.hidden = True
    remaining = [t for t in tokens if t != hidden_token]

    token_seq, delta_seq = [], []
    current_token = rng.choice(remaining)
    for counter in range(seq_length - 1):
        token_seq.append(current_token)
        next_token = rng.choice(remaining)
        if counter < seq_length - 2:
            delta = (next_token.coordinates[0] - current_token.coordinates[0],
                     next_token.coordinates[1] - current_token.coordinates[1])
            delta_seq.append(delta)
            current_token = next_token

    last_token = token_seq[-1]
    token_seq += [hidden_token, last_token]
    delta_seq.append((hidden_token.coordinates[0] - last_token.coordinates[0],
                      hidden_token.coordinates[1] - last_token.coordinates[1]))
    last_delta = (-delta_seq[-1][0], -delta_seq[-1][1])
    delta_seq.append(last_delta)
    return token_seq, delta_seq



def generate_forbidden_transition(world, sequence_length, n = 30, start_token=None, rng=None, max_tries=100,):
    """
    Generates a token sequence of length (sequence_length + 1) and a displacement
    sequence of length sequence_length.

    Constraints:
      - The transition token1 -> token2 occurs exactly once, at step n
        (i.e., token_sequence[n-1] == token1 and token_sequence[n] == token2).
      - The transition token1 -> token2 does not occur at any other step.
      - No self-transitions (next_token != current_token), matching your original logic.
    """
    rng = rng or world.rng

    token_sequence, delta_seq = [], []

    token_zero = next(t for t in world.tokens if t.coordinates == (0, 0))
    current_token = start_token if start_token is not None else token_zero

    token_sequence.append(current_token)
    token1 = random.choice(world.tokens)
    token1.hidden = True

    token2 = random.choice(list(set(world.tokens)-{token1}))
    token2.hidden = True

    forbidden = {(token1, token2), (token2, token1)}


    for step in range(sequence_length):
        if step == n - 2:
            # put sequence on a non-hidden token, different from current
            choices = [t for t in world.tokens if (t not in (token1, token2)) and (t != current_token)]
            if not choices:
                raise RuntimeError("No valid token to place at step n-2 (world too small).")
            next_token = rng.choice(choices)

        elif step == n - 1:
            # now current_token is guaranteed non-hidden, so this edge cannot be forbidden
            next_token = rng.choice((token1, token2))

        elif step == n:
            # force the forbidden edge exactly here
            next_token = token2 if current_token == token1 else token1

        #normal sequence without forbidden transition
        else:
            tries = 0
            while True:
                tries += 1
                if tries > max_tries:
                    raise RuntimeError(
                        f"Exceeded max_tries={max_tries} while sampling step={step}. "
                        "World may be too small or constraints too tight."
                    )

                candidate = world.get_random_token(rng=rng)

                if candidate == current_token:
                    continue
                if current_token == token1 and candidate == token2:
                    continue
                if current_token == token2 and candidate == token1:
                    continue

                next_token = candidate
                break

        delta = (
            next_token.coordinates[0] - current_token.coordinates[0],
            next_token.coordinates[1] - current_token.coordinates[1],
        )
        delta_seq.append(delta)
        token_sequence.append(next_token)
        current_token = next_token

    return token_sequence, delta_seq


def generate_world_seq(labels, grid_size, tokens_amount, sequence_length, *, seq_type="gaze", min_distance=0.25, batch_size=1, no_repeats=False, rng=None, policy_kwargs=None, put_k=None):
    rng = rng or np.random.default_rng()
    policy_kwargs = policy_kwargs or {}

    world_sequence_map = {}

    for _ in range(batch_size):

        fixed_k = False
        if seq_type == "fixed_k":
            fixed_k = True

        world = TokenWorld(
            labels,
            grid_size,
            tokens_amount,
            min_distance,
            no_repeats,
            rng=rng,
            fixed_k=fixed_k,
            put_k=put_k,
        )

        if seq_type in ("gaze", "fixed_k"):
            token_seq, direction_seq = generate_gaze_sequence(
                world, sequence_length, rng=rng
            )

        elif seq_type == "hidden":
            token_seq, direction_seq = generate_hidden_token_seq( world, sequence_length, rng=rng)
        
        elif seq_type == "transition":
            token_seq, direction_seq = generate_forbidden_transition(world,sequence_length, ** policy_kwargs)

        else:
            raise ValueError(f"Unknown seq_type '{seq_type}'")

        world_sequence_map[world] = (token_seq, direction_seq)

    return world_sequence_map

def add_token_to_seq(tokens_seq, directions_seq, new_token):
    last_token = tokens_seq[-1]
    tokens_seq.append(new_token)
    delta = (new_token.coordinates[0] - last_token.coordinates[0], new_token.coordinates[1] - last_token.coordinates[1])
    directions_seq.append(delta)
    # check_sequence_direction(tokens_seq, directions_seq, len(directions_seq)-1, len(directions_seq))
    return tokens_seq, directions_seq

def expand_sequence(world_seq, add_amount, changed_token_map=None, rng=None):
    rng = rng or np.random.default_rng()

    for world, (tokens_seq, directions_seq) in world_seq.items():
        tokens = list(world.tokens)
        skip = None

        if changed_token_map:
            if world in changed_token_map:
                skip = changed_token_map[world]
            else:
                for w_key, tok in changed_token_map.items():
                    if id(w_key) == id(world):
                        skip = tok
                        break

        if skip:
            old_len = len(tokens)
            tokens = [t for t in tokens if t.label != skip.label]
            new_len = len(tokens)
        #     print(f"[expand_sequence] Skipping label '{skip.label}' "
        #           f"(removed {old_len - new_len}) in world {id(world)}")

        # print(f"[expand_sequence] Candidates: {[t.label for t in tokens]}")

        current_token = tokens_seq[-1]
        for _ in range(add_amount):
            new_token = rng.choice(tokens)
            while new_token is current_token:
                new_token = rng.choice(tokens)

            add_token_to_seq(tokens_seq, directions_seq, new_token)
            current_token = new_token

    return world_seq


def pick_next_token(rng, candidates, current_token):
    """Pick a token != current_token."""
    nxt = rng.choice(candidates)
    while nxt is current_token:
        nxt = rng.choice(candidates)
    return nxt



def save_list(data, graphs_dir, filename):
    os.makedirs(graphs_dir, exist_ok=True)
    path = os.path.join(graphs_dir, f"{filename}.pkl")
    with open(path, "wb") as f:
        pkl.dump(data, f)
    print(f"List saved to {path}")
