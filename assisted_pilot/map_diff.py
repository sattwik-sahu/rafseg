import numpy as np

def process_attention_maps(go_to_map, avoid_map):
    # Normalize both maps to 0-1 range
    go_to_norm = go_to_map.astype(float) / 255
    avoid_norm = avoid_map.astype(float) / 255

    # Invert the avoid map
    avoid_inverted = 1 - avoid_norm

    # Multiply go_to map with inverted avoid map
    result = go_to_norm * avoid_inverted

    # Scale back to 0-255 range and convert to uint8
    result_uint8 = (result * 255).astype(np.uint8)

    return result_uint8



def process_attention_maps_weighted(go_to_map, avoid_map, go_to_weight=1.0, avoid_weight=1.0):
    # Ensure weights are positive
    go_to_weight = max(0, go_to_weight)
    avoid_weight = max(0, avoid_weight)

    # Normalize both maps to 0-1 range
    go_to_norm = go_to_map.astype(float) / 255
    avoid_norm = avoid_map.astype(float) / 255

    # Apply weights
    go_to_weighted = go_to_norm * go_to_weight
    avoid_weighted = avoid_norm * avoid_weight

    # Clip weighted maps to 0-1 range
    go_to_weighted = np.clip(go_to_weighted, 0, 1)
    avoid_weighted = np.clip(avoid_weighted, 0, 1)

    # Invert the weighted avoid map
    avoid_inverted = 1 - avoid_weighted

    # Multiply go_to map with inverted avoid map
    result = go_to_weighted * avoid_inverted

    # Scale back to 0-255 range and convert to uint8
    result_uint8 = (result * 255).astype(np.uint8)

    return result_uint8

