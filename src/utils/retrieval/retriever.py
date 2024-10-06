from abc import ABC, abstractmethod
from typing import List, Callable
import numpy as np
from numpy import typing as npt
from utils.vision.embeddings.vector_store import VectorStore, DocumentVector


class Retriever[T_VectorStore: VectorStore, T_Document: DocumentVector](ABC):
    """
    Retriever class for vector retrieval
    """

    _vector_store: T_VectorStore
    _similarity: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]

    def __init__(
        self,
        vector_store: T_VectorStore,
        sim_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    ) -> None:
        self._vector_store = vector_store
        self._similarity = sim_func

    @abstractmethod
    def __call__(
        self, query_vector: npt.NDArray[np.float64], *args, **kwargs
    ) -> List[T_Document]:
        pass
