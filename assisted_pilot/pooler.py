import numpy as np
import cv2
import torch
import typing as t

class Pooler():
    def __init__(self, pooling_method: t.Literal['max', 'avg']) -> None:
        self.pooling_method = pooling_method

    def pool(self, maps: np.ndarray) -> np.ndarray:
        """
        Pool the attention maps and return a single pooled map.
        """
        pooled_map = np.zeros_like(maps[0])
        
        if self.pooling_method == 'max':
            pooled_map = np.max(maps, axis=0)
        elif self.pooling_method == 'avg':
            pooled_map = np.mean(maps, axis=0)
        else:
            raise ValueError("Pooling method must be either 'max' or 'avg'.")
        
        return pooled_map