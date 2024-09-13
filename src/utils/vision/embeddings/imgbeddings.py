import typing as t
from pathlib import Path

from imgbeddings import imgbeddings as Imgbeddings
from numpy import ndarray
from PIL.Image import Image
from torch._tensor import Tensor
from typing_extensions import override

from utils.vision.embeddings.base import EmbeddingPipeline


class ImgbeddingsPipeline(EmbeddingPipeline[Imgbeddings, Image]):
    """
    The Imgbeddings image embedding pipeline.
    """

    img_cache_path: str | Path

    def __init__(self, model: Imgbeddings, img_cache_path: str | Path) -> None:
        super().__init__(model)
        self.img_cache_path = img_cache_path

    @override
    def _run_model(
        self, x: Image | t.List[Image]
    ) -> ndarray | Tensor | t.List[float] | t.List[t.List[float]]:
        return self.model.to_embeddings(inputs=x)
