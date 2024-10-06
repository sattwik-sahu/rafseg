import typing as t
from pathlib import Path

from numpy import ndarray
from PIL.Image import Image
from torch._tensor import Tensor
from typing_extensions import override

from utils.vision.embeddings.base import EmbeddingPipeline
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch

class DinoPipeline(EmbeddingPipeline):
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        

    @override
    def _run_model(
        self, x: Image | t.List[Image]
        ) -> ndarray | Tensor | t.List[float] | t.List[t.List[float]]:

        inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state.mean(dim=1)

        return last_hidden_states
    
    @override
    def _postprocess(self, y: ndarray | Tensor | t.List[float] | t.List[t.List[float]], *args: t.Any, **kwargs: t.Any) -> ndarray:
        if y.device.type == 'cuda':
            return y.cpu().detach().numpy()
        
        if isinstance(y, np.ndarray):
            return y
        
        return np.array(y)