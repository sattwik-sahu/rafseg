import numpy as np
from dataclasses import dataclass
import typing as t
from abc import ABC, abstractmethod
import torch
from pathlib import Path
import pickle


@dataclass
class DocumentVector:
    id: int
    embedding: np.ndarray


T_Document = t.TypeVar("T_Document")


class VectorStore[T_Document: DocumentVector](ABC):
    def __init__(self) -> None:
        self._documents: t.List[T_Document] = []
        self._vectors: np.ndarray = np.empty(0)

    def add(self, doc: T_Document) -> None:
        self._documents.append(doc)
        if self._vectors.size == 0:
            self._vectors = doc.embedding
        else:
            self._vectors = np.vstack((self._vectors, doc.embedding))

    @abstractmethod
    def _get_similarity_scores(
        self, query_vec: np.ndarray
    ) -> np.ndarray | torch.Tensor:
        """
        Calculates the similarity metric between the query vector `query_vec`
        all the different vectors stored in the index.

        Args:
            query (np.ndarray): The query vector. Should have dim `(1, n_dim)` where
                `n_dim` is the embedding vector size.

        Returns:
            np.ndarray: An array of shape `(n_vecs,)` giving a single scalar value for
                each of the `n_vecs` vectors stored in the index.
        """
        pass

    def retrieve(self, query_embedding: np.ndarray, k: int) -> t.List[DocumentVector]:
        """
        Calculates similarity to all vectors in the index and returns the
        `k` most similar vectors' corresponding documents.

        Args:
            query_embedding (np.ndarray): The query embedding.
            k (int): How many of the top most similar documents to retrieve?

        Returns:
            t.List[DocumentVector]: The list of `k` documents with the most
                similar embeddings.
        """
        # This gets the indexes of the top k similar embeddings and
        # gets those indexes from the documents array
        return self._documents[
            np.argpartition(self._get_similarity_scores(query_vec=query_embedding), -k)[
                :-k
            ]
        ]

    @property
    def documents(self) -> t.List[T_Document]:
        return self._documents

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file=file)
