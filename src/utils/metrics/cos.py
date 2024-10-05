import numpy as np


def cosine_similarity(arr1: np.ndarray, arr2: np.ndarray) -> float:
    numerator = np.dot(arr1, arr2)
    denominator = np.linalg.norm(arr1) * np.linalg.norm(arr2) + 0.0001

    return numerator / denominator
