import random
import math
import string
from token_world import TokenInstance, add_token_to_seq, expand_sequence, generate_world_seq, save_list


LETTERS = list(string.ascii_lowercase)

"""change labels """
def change_label(world_maps, pos, continued_len):
    changed_maps = {}
    changed_token_map = {}

    for world, (token_seq, directions_seq) in world_maps.items():
        new_token_seq = token_seq[:pos]
        new_directions_seq = directions_seq[:(pos-1)]

        # choose token to "relabel"
        chosen_token = random.choice(world.tokens)
        used_labels = [t.label for t in world.tokens]
        free_labels = list(set(LETTERS) - set(used_labels))
        new_label = random.choice(free_labels)

        # create new token (temporal fork)
        new_token = TokenInstance(new_label, chosen_token.coordinates, LETTERS)
        new_token.old_label = chosen_token.label

        # modify the world:
        world.tokens.append(new_token)
        # mark which token must NOT be used after this point
        changed_token_map[world] = chosen_token

        # extend sequence with the new token
        add_token_to_seq(new_token_seq, new_directions_seq, new_token)

        changed_maps[world] = (new_token_seq, new_directions_seq)
        
    # expansion will avoid the old token entirely
    expand_sequence(changed_maps, continued_len, changed_token_map)
    
    return changed_maps



"""pentagon-shaped world"""
def pentagon_shape(world_maps):
    radius = 3.0  
    center = (0, 0)

    # 5 outer points of pentagon
    pentagon_positions = [
        (
            center[0] + radius * math.cos(2 * math.pi * i / 5),
            center[1] + radius * math.sin(2 * math.pi * i / 5)
        )
        for i in range(5)
    ]

    # Add center point
    pentagon_positions.append(center)

    for world, (token_seq, directions_seq) in world_maps.items():
        for token, position in zip(world.tokens, pentagon_positions):
            token.coordinates = position


def move_k(world_maps):
    for world in world_maps.keys():
        for t in world.tokens:
            if t.label =="k":
                searching = True
                while searching:
                    x = random.uniform(-4, 4)
                    y = random.uniform(-4, 4)
                    if not (x > 0 and y > 0):
                        searching = False
                        t.coordinates = (x,y)


def change_loc_token(world_maps):
    for world in world_maps.keys():
        not_k_list = [t for t in world.tokens if t.label != "k"]
        selected_token = random.choice(not_k_list)
        selected_token.coordinates = (1.0, 1.0)
        
if __name__ == "__main__":

    seq_len = 0
    continued_len = 100

    title = f"pentagon_worlds"
    world_maps = generate_world_seq(LETTERS, (-4,4), (6,6), sequence_length=seq_len,  batch_size=1000, seq_type="gaze")
    # world_maps = change_label(world_maps, seq_len, continued_len)
    # expand_sequence(world_maps, continued_len)
    pentagon_shape(world_maps)
    expand_sequence(world_maps, continued_len)
    save_list(world_maps, "paper_data/sequences", title)