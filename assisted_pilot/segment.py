import cv2
import numpy as np
import torch
import typing as t

class Segmenter():
    def __init__(self, method: t.Literal['static', 'otsu']) -> None:
        self.method = method

    def threshold(self, image: np.ndarray, low: int = 0, high: int = 255) -> np.ndarray:
        if self.method == 'static':
            _, thresh = cv2.threshold(image, low, high, cv2.THRESH_BINARY)
        elif self.method == 'otsu':
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError("Method must be either 'static' or 'otsu'.")
        return thresh

    def segment(self, image: np.ndarray, low: int = 0, high: int = 255) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask =  self.threshold(gray, low, high)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask

        