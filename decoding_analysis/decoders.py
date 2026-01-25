import numpy as np
from sklearn.calibration import LabelEncoder, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import torch


def svm_label_decoder(
    hidden_states,          # list/array/tensor of shape (W,T,H) OR list of (T,H)
    token_seq_list,         # list length W, each is seq of token objects with .label
    cv_folds,
    target_offset=0,        # off
    min_t=30,               # pos_start
    max_t=99,              # pos_stop (exclusive)
    seed=0,
    max_iter=2000,
):
    # --- A: (W,T,H) float32 numpy ---
    if torch.is_tensor(hidden_states):
        A = hidden_states.detach().cpu().numpy()
    else:
        A = np.asarray(hidden_states)

    # If passed as list of (T,H), stack to (W,T,H)
    if A.dtype == object or A.ndim == 2:
        # object array or single (T,H): normalize to list then stack
        hs_list = hidden_states if isinstance(hidden_states, (list, tuple)) else [hidden_states]
        hs_list = [(h.detach().cpu().numpy() if torch.is_tensor(h) else np.asarray(h)) for h in hs_list]
        A = np.stack(hs_list, axis=0)

    # Handle accidental (W,1,T,H) or (1,T,H)
    if A.ndim == 4 and A.shape[1] == 1:
        A = A[:, 0]
    if A.ndim == 3 and A.shape[0] == 1 and len(token_seq_list) != 1:
        # unusual; leave as-is unless you know you have a singleton batch
        pass

    A = A.astype(np.float32, copy=False)   # (W,T,H)
    W, T, H = A.shape

    # --- Y: (W,T) int labels, globally encoded like decode_timepoint_tokens ---
    # Build raw labels aligned to timepoints 0..T-1 (clamped by available seq length)
    raw = []
    for seq in token_seq_list:
        L = min(len(seq), T)
        raw.append([seq[t].label for t in range(L)] + [None] * (T - L))

    # Fit encoder on all non-None labels (global mapping)
    flat_labels = [lab for row in raw for lab in row if lab is not None]
    if len(set(flat_labels)) < 2:
        raise ValueError(f"Only one class found in labels: {set(flat_labels)}. Cannot train classifier.")

    enc = LabelEncoder().fit(flat_labels)

    Y = np.full((W, T), -1, dtype=np.int64)
    for i, row in enumerate(raw):
        valid = [lab for lab in row if lab is not None]
        if valid:
            Y[i, :len(valid)] = enc.transform(valid)

    # --- bounds exactly like decode_timepoint_tokens ---
    off = int(target_offset)
    p_lo = max(min_t, -off)
    p_hi_excl = min(max_t, T - max(off, 0))  # exclusive

    if p_hi_excl <= p_lo:
        raise ValueError(f"No valid positions for offset={off}: p_lo={p_lo}, p_hi_excl={p_hi_excl}, T={T}")

    X = A[:, p_lo:p_hi_excl, :].reshape(-1, H)
    y = Y[:, (p_lo + off):(p_hi_excl + off)].reshape(-1)

    # Sanity: y should have no -1 if lengths match; if it does, you had shorter seqs than T
    if np.any(y < 0):
        raise ValueError("Found invalid label entries (-1). Sequence lengths likely shorter than hidden-state T.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
    )

    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    return {
        "acc_mean": scores.mean(),
        "acc_std": scores.std(ddof=0),
        "acc_scores": scores,
        "used_p_lo": int(p_lo),
        "used_p_hi_excl": int(p_hi_excl),
        "N": int(X.shape[0]),
    }


def tuple_decoder( hidden_states, world_seqs, tuples_dict=None, 
                  coord_offset=0, label_offset=0, cv_folds=5, min_t=30, max_t=None, mode="tuple", pos_id=None):

    """
    Decode (label, coordinate) tuples from hidden states using a linear SVM.
    Each sample corresponds to a hidden state at timestep t, labeled by
    (label[t + label_offset], position[t + coord_offset]).

    Parameters
    ----------
    hidden_states : list[np.ndarray] Hidden state tensors for each world (shape [T, D]).
    world_seqs : dict Mapping world_id -> (tokens, world_meta).
    tuples_dict : dict Mapping (label, (x, y)) -> tuple ID.
    coord_offset, label_offset : int Temporal offsets for coordinate and label indices.
    cv_folds : int Number of stratified CV folds.
    min_t, max_t : Optional[int] Temporal window for sampling timesteps.
    """

    X_list, y = [], []

    for hidden_tensor, (tokens, _) in zip(hidden_states, world_seqs.values()):
        h_arr = np.asarray(hidden_tensor)

        # range of valid time indices given offsets
        max_step = max_t if max_t is not None else len(tokens) - (1 + max(abs(coord_offset), abs(label_offset)))
        min_step = min_t if min_t is not None else max(abs(coord_offset), abs(label_offset))
        for t in range(min_step, max_step):
            
            if mode == "position":
                coord = tokens[t + coord_offset].coordinates
                pos_key = (round(coord[0], 4), round(coord[1], 4))

                if pos_id is None:
                    raise ValueError("pos_id must be provided when mode='position'")

                if pos_key not in pos_id:
                    print(f"key {pos_key} not in dict")
                    continue

                target = pos_id[pos_key]

            elif mode == "tuple":
                label = tokens[t + label_offset].label
                coord = tokens[t + coord_offset].coordinates
                key = (label, (round(coord[0], 4), round(coord[1], 4)))

                if key not in tuples_dict:
                    continue

                target = tuples_dict[key]

            else:
                raise ValueError(f"Unknown mode: {mode}")

            X_list.append(h_arr[t])
            y.append(target)

    if not X_list:
        return {"acc_mean": np.nan, "acc_std": np.nan, "acc_scores": []}

    X = np.vstack(X_list)
    y = LabelEncoder().fit_transform(y)
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print("starting validation")
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=1)    
    print("finished validation")
    
    return {
        "acc_mean": float(np.mean(scores)),
        "acc_std": float(np.std(scores)),
        "acc_scores": scores,
    }
