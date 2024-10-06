from utils.retrieval.retriever import Retriever
from utils.vision.embeddings.vector_store import VectorStore, DocumentVector
from typing import Callable, List
from numpy import typing as npt
import numpy as np
from typing_extensions import override


class MaximumMarginalRelevanceRetriever[
    T_VectorStore: VectorStore,
    T_Document: DocumentVector,
](Retriever[T_VectorStore, T_Document]):
    """
    Retrieval by Maximum Marginal Relevance.
    """

    # _k: int
    _lambda: float
    _vs_max_sims: List[float]
    _vecs: npt.NDArray[np.float64]

    def __init__(
        self,
        vector_store: T_VectorStore,
        # k: int,
        param_lambda: float,
        sim_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    ) -> None:
        super().__init__(vector_store=vector_store, sim_func=sim_func)
        # self._k = k
        self._lambda = param_lambda
        self._vecs = self._vector_store.vectors
        self._vs_max_sims = []
        self._precalc_stored_vector_max_sims()

    def _precalc_stored_vector_max_sims(self) -> None:
        print("Precalculating max similarities...")
        for i, vec in enumerate(self._vecs):
            self._vs_max_sims.append(
                np.max(
                    [
                        self._similarity(vec, vec_)
                        for j, vec_ in enumerate(self._vecs)
                        if j != i
                    ]
                )
            )
            print(f"Done {i + 1}/{self._vecs.shape[0]}")

    @override
    def __call__(
        self, query_vector: npt.NDArray[np.float64], k: int
    ) -> List[T_Document]:
        mmr_scores: List[float] = []
        for i, vec in enumerate(self._vecs):
            mmr_scores.append(
                self._lambda * self._similarity(query_vector.ravel(), vec)
                - (1 - self._lambda) * self._vs_max_sims[i]
            )
        k_best_mmr_inx: npt.NDArray[np.int32] = np.argsort(mmr_scores)[-k:]
        return [self._vector_store.documents[i] for i in k_best_mmr_inx]
