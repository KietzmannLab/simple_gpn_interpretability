import numpy as np
from utils import load_world_sequences


def check_hidden_token_rules(tokens_seq, world, switch_idx):
    token_seq_labels = [tok.label for tok in tokens_seq]
    new = None

    for token in world.tokens: 
        if token.hidden: 
            new = token.label

    
    before = token_seq_labels[:switch_idx]
    after = token_seq_labels[switch_idx:]


    assert new is not None, f"no hidden token is present"
    assert new not in before, f"❌ Hidden token '{new}' found before switch index {switch_idx}."
    assert new in after, f"❌ Hidden token '{new}' not found after switch index {switch_idx}."

    print(f"✅ Passed check: hidden '{new}' appears only after index {switch_idx}.")


def check_forbidden_transition(token_seq, world, transition_at):
    hidden = [t for t in world.tokens if t.hidden]

    a, b = hidden
    forbidden = {(a, b), (b, a)}

    # build all edges i: token_seq[i] -> token_seq[i+1]
    edges = [(token_seq[i], token_seq[i + 1]) for i in range(len(token_seq) - 1)]
    bad_idxs = [i for i, e in enumerate(edges) if e in forbidden]

    # Debug prints
    print("Hidden tokens:", a.label, b.label)
    if 0 <= transition_at < len(edges):
        u, v = edges[transition_at]
        print(f"Edge at {transition_at}: {u.label} -> {v.label}",
              "| forbidden?" , (u, v) in forbidden)
    print("All forbidden edge indices:", bad_idxs)
    if bad_idxs:
        print("Forbidden edges (first few):",
              [(i, edges[i][0].label, edges[i][1].label) for i in bad_idxs[:10]])

    # Assertions matching the intent
    assert 0 <= transition_at < len(edges), "transition_at out of range"
    assert edges[transition_at] in forbidden, f"No forbidden transition at {transition_at}"
    assert bad_idxs == [transition_at], f"Forbidden transitions at {bad_idxs}, expected only [{transition_at}]"

    print("✅ Passed check: forbidden transition occurs exactly once at the expected step.")

def check_decay( tokens_seq, world, *, pre_steps, gap_steps, reintro_step=None, expect_pre=1, verbose=True):
    
    """
    Validate decay sequence structure.

    Checks:
      - hidden appears exactly `expect_pre` times during pre-exposure
      - hidden appears 0 times during gap
      - hidden appears exactly once at reintroduction step
      - reports how many times it appears after reintroduction
    """
    # --- identify hidden token ---
    hidden_tokens = [t for t in world.tokens if t.hidden]
    assert len(hidden_tokens) == 1, "❌ Expected exactly one hidden token."

    hidden_label = hidden_tokens[0].label
    labels = [tok.label for tok in tokens_seq]

    if reintro_step is None:
        reintro_step = pre_steps + gap_steps

    # --- find all indices ---
    all_indices = [i for i, lab in enumerate(labels) if lab == hidden_label]

    # --- phase indices ---
    pre_indices   = [i for i in all_indices if i < pre_steps]
    gap_indices   = [i for i in all_indices if pre_steps <= i < reintro_step]
    reintro_hits  = [i for i in all_indices if i == reintro_step]
    post_indices  = [i for i in all_indices if i > reintro_step]

    # --- assertions ---
    assert len(pre_indices) == expect_pre, (
        f"❌ Hidden '{hidden_label}' appears {len(pre_indices)} times in pre phase, "
        f"expected exactly {expect_pre}. Indices: {pre_indices}"
    )

    assert len(gap_indices) == 0, (
        f"❌ Hidden '{hidden_label}' appears during gap at indices {gap_indices}."
    )

    # assert len(reintro_hits) == 1, (
    #     f"❌ Hidden '{hidden_label}' reintro count = {len(reintro_hits)}, "
    #     f"indices = {reintro_hits}"
    # )

    if verbose:
        print(f"🔎 Hidden token '{hidden_label}' occurrence indices:")
        print(f"   pre-phase:      {pre_indices}")
        print(f"   gap-phase:      {gap_indices}")
        print(f"   reintro-step:   {reintro_hits}")
        print(f"   post-phase:     {post_indices}")
        print(f"   total count:    {len(all_indices)}")

    return {
        "hidden_label": hidden_label,
        "all_indices": all_indices,
        "pre_indices": pre_indices,
        "gap_indices": gap_indices,
        "reintro_indices": reintro_hits,
        "post_indices": post_indices,
    }

def check_label_change_rules(tokens_seq, world):
    
    changed = next((t for t in world.tokens if getattr(t, "old_label", None)), None)
    assert changed is not None, "❌ No changed token found (missing old_label)."

    old = changed.old_label
    new = changed.label

    token_seq_labels = [tok.label for tok in tokens_seq]

    assert new in token_seq_labels, f"❌ New label '{new}' never appears in sequence."
    assert old in token_seq_labels, f"❌ Old label '{old}' never appears in sequence."

    # define switch as first appearance of new
    switch_idx = token_seq_labels.index(new)

    before = token_seq_labels[:switch_idx]
    after  = token_seq_labels[switch_idx:]

    assert new not in before, f"❌ New label '{new}' appears before switch_idx={switch_idx}."
    assert old not in after,  f"❌ Old label '{old}' appears after switch_idx={switch_idx}."

    print(f"✅ Passed: old '{old}' only before {switch_idx}, new '{new}' from {switch_idx} onward.")


def check_sequence_direction(tokens_seq, dirs_seq, idx_a=5, idx_b=6, label="", atol=1e-6):
    """
    Compute ΔX, ΔY and verify that it matches the recorded direction vector.
    """
    a, b = tokens_seq[idx_a], tokens_seq[idx_b]
    dy = b.coordinates[1] - a.coordinates[1]  # y = 2nd coordinate
    dx = b.coordinates[0] - a.coordinates[0]  # x = 1st coordinate

    delta = np.array([dx, dy])
    # print(f"[{label}] ΔX={dx:.2f}, ΔY={dy:.2f}")

    if dirs_seq:
        recorded = np.array(dirs_seq[idx_a])
        # print(f"[{label}] Direction in sequence: {recorded}")
        assert np.allclose(delta, recorded, atol=atol), (
            f"Mismatch between computed Δ({delta}) and recorded direction {recorded} "
            f"at indices {idx_a}->{idx_b}"
        )
    else:
        print(f"[{label}] Direction in sequence: None")

    return delta

def check_fixed_k(world, center=(1.0, 1.0), min_dist=0.25):
    center = np.array(center, dtype=float)

    # collect k tokens
    k_tokens = [t for t in world.tokens if t.label == "k"]

    if len(k_tokens) > 0:
        # must be exactly one k, exactly at the center
        assert len(k_tokens) == 1, f"Expected exactly 1 'k', found {len(k_tokens)}"
        k = k_tokens[0]
        assert tuple(k.coordinates) == tuple(center), f"'k' must be at {tuple(center)}, got {k.coordinates}"

        # optional: enforce other tokens are still >= min_dist away from center (excluding k itself)
        for t in world.tokens:
            if t is k:
                continue
            d = np.linalg.norm(np.array(t.coordinates, dtype=float) - center)
            assert d >= min_dist, f"Token {t.label}@{t.coordinates} too close to center: dist={d:.3f} < {min_dist}"
    else:
        # no k: then no token can be within min_dist of center
        for t in world.tokens:
            d = np.linalg.norm(np.array(t.coordinates, dtype=float) - center)
            assert d >= min_dist, f"No 'k' in world, but {t.label}@{t.coordinates} is too close: dist={d:.3f} < {min_dist}"
    print(len(k_tokens))


def check_moved_k(world, center=(1.0, 1.0), min_dist=0.25):
    center = np.array(center, dtype=float)

    # collect k tokens
    k_tokens = [t for t in world.tokens if t.label == "k"]
    one_tokens = [t for t in world.tokens if t.coordinates == (1,1)]

    if len(k_tokens) > 0:
        # must be exactly one k, exactly at the center
        assert len(k_tokens) == 1, f"Expected exactly 1 'k', found {len(k_tokens)}"
        k = k_tokens[0]
        assert tuple(k.coordinates) != tuple(center), f"'k' must be at {tuple(center)}, got {k.coordinates}"
    else:
        # no k: then no token can be within min_dist of center
        for t in world.tokens:
            d = np.linalg.norm(np.array(t.coordinates, dtype=float) - center)
            assert d >= min_dist, f"No 'k' in world, but {t.label}@{t.coordinates} is too close: dist={d:.3f} < {min_dist}"
    
    assert len(one_tokens) == 1, f"no token in the right place {len(one_tokens)}"




def inspect_world_sequences(path, changed_at=31):
    """
    Inspect a saved world sequence file (e.g., line worlds or label-change sets)
    for coordinate consistency and label switching rules.
    """
    new_seqs = load_world_sequences(path)
    print(f"Loaded {len(new_seqs)} worlds from {path}\n")

    for world, (tokens_seq, dirs_seq) in new_seqs.items():
        print(f"World Tokens: {[(t.label, t.coordinates) for t in world.tokens]}")

        for t in world.tokens:
            if t.hidden:
                print(f"Hidden token: {t.label}")
                check_hidden_token_rules(tokens_seq, t, changed_at)

            if getattr(t, "old_label", None):
                print(f"Changed token: {t.label}:{t.old_label}")
                check_label_change_rules(tokens_seq, t)

           # modular direction checks
        check_sequence_direction(tokens_seq, dirs_seq, 5, 6, "Segment 1")
        check_sequence_direction(tokens_seq, dirs_seq, -2, -1, "Segment last")


def analyze_line_alignment(world, tol_deg=1.5):
    """
    Analyze whether tokens in a given world lie on a line,
    and compute their mean orientation and spacing.

    Parameters
    ----------
    world : World
        World object with .tokens having .coordinates
    tol_deg : float
        Maximum angular deviation (for classification as aligned)

    Returns
    -------
    dict
        {
            'mean_angle': float,
            'mean_spacing': float,
            'aligned': bool,
            'n_tokens': int
        }
    """
    coords = np.array([t.coordinates for t in world.tokens])
    if len(coords) < 2:
        return {"mean_angle": np.nan, "mean_spacing": np.nan, "aligned": False, "n_tokens": len(coords)}

    deltas = np.diff(coords, axis=0)
    mean_dir = np.mean(deltas, axis=0)
    mean_angle = np.degrees(np.atan2(mean_dir[1], mean_dir[0]))
    spacing = np.mean(np.linalg.norm(deltas, axis=1))

    # optional: deviation from a perfectly straight line
    dir_norms = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
    ang_devs = np.degrees(np.arccos(np.clip(dir_norms @ dir_norms[0], -1, 1)))
    max_dev = np.max(np.abs(ang_devs))

    return {
        "mean_angle": mean_angle,
        "mean_spacing": spacing,
        "aligned": max_dev < tol_deg,
        "max_deviation": max_dev,
        "n_tokens": len(coords),
    }

def inspect_angled_sequences(path, n_samples=3, tol_deg=1.5):
    sequences_dict = load_world_sequences(path)
    print(f"Loaded {len(sequences_dict)} models from {path}")

    for model_name, angles_dict in sequences_dict.items():
        print(f"\n=== Model {model_name} ===")
        printed = 0

        for angle_offset, seqs in angles_dict.items():
            if printed >= n_samples:
                break

            world = next(iter(seqs.keys()))  # take the first world
            metrics = analyze_line_alignment(world, tol_deg=tol_deg)
            mean_angle = metrics["mean_angle"]
            deviation = np.abs((mean_angle - angle_offset + 180) % 360 - 180)

            print(f"Angle offset = {angle_offset:+.1f}° → Measured mean = {mean_angle:+.2f}° "
                  f"(Δ={deviation:.2f}°, max dev={metrics['max_deviation']:.2f}°)")
            print(f"Tokens: {metrics['n_tokens']} | spacing ≈ {metrics['mean_spacing']:.3f}")
            print(f"✓ {'Aligned' if metrics['aligned'] else '⚠ Misaligned'} (tol ±{tol_deg}°)")

            printed += 1

    print("\nInspection complete.")

if __name__ == "__main__":
    tol_deg = 1.5
    n_samples = 5   # number of worlds to print per model
    #angled_sequences_2m_1112_1556
    #pentagon_worlds_internship
    #changed_30_6_sequences_internship
    #expanded_6_sequences_internship
    #changed_labels_30_100
    #changed_30_6_sequences_internship
    #decay_1_reintro_sequences 

    world_seqs = load_world_sequences("paper_data/sequences/change_at_35.pkl")
    print(f"Loaded {len(world_seqs)} sequences ")

    summary = {}
    for world, (tok_seq, dir_seq) in world_seqs.items():
        for i in range(len(tok_seq) - 1):
            check_sequence_direction(tok_seq, dir_seq, i, i+1)
        # check_forbidden_transition(token_seq=tok_seq, world=world, transition_at=100)

        # check_hidden_token_rules(tok_seq, world, 34)
        check_label_change_rules(tok_seq, world)
        # check_decay(tokens_seq=tok_seq,world=world,pre_steps=21, gap_steps=50, reintro_step=70, expect_pre=1)
        # check_fixed_k(world)

    # for model_name, angles_dict in world_seqs.items():
    #     print(f"\n=== Model {model_name} ===")
    #     summary[model_name] = {}

    #     for angle_offset, seqs in angles_dict.items():
    #         results = []
    #         for world, (tok_seq, dir_seq) in seqs.items():
    #             check_sequence_direction(tok_seq, dir_seq,0,1) 
    #             check_sequence_direction(tok_seq, dir_seq,len(dir_seq)-1,len(dir_seq)) 
    #             metrics = analyze_line_alignment(world, tol_deg=tol_deg)
    #             deviation = np.abs((metrics["mean_angle"] - angle_offset + 180) % 360 - 180)
    #             metrics["deviation"] = deviation
    #             results.append(metrics)

    #         # aggregate across worlds for this angle
    #         mean_dev = np.nanmean([r["deviation"] for r in results])
    #         aligned_ratio = np.mean([r["aligned"] for r in results])
    #         summary[model_name][angle_offset] = {
    #             "mean_deviation": mean_dev,
    #             "aligned_ratio": aligned_ratio,
    #             "n_worlds": len(results),
    #         }

    #         # print a few examples for verification
    #         if np.random.rand() < (n_samples / len(angles_dict)):
    #             print(f"\nAngle {angle_offset:+.1f}° → mean deviation {mean_dev:.2f}°, "
    #                   f"{aligned_ratio*100:.1f}% aligned (n={len(results)})")

    # print("\n=== Summary snapshot ===")
    # print({k: list(v.keys())[:3] for k, v in summary.items()})
    print("\nInspection complete.")