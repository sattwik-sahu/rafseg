import typing as t
from abc import ABC, abstractmethod

import numpy as np
import torch

T_Model: t.TypeVar
T_Input: t.TypeVar


class EmbeddingPipeline[T_Model, T_Input](ABC):
    """
    Class to create a pipeline for embedding images.
    """

    model: T_Model

    def __init__(self, model: T_Model, *args, **kwargs) -> None:
        """
        Initializes an EmbeddingPipeline object.

        Args:
            model (T_Model): The embedding model.
        """
        self.model = model

    def _preprocess(
        self, x: T_Input | t.List[T_Input], *args: t.Any, **kwargs: t.Any
    ) -> T_Input | t.List[T_Input]:
        """
        Performs preprocessing on the input before passing through the
        embedding pipeline.

        Override this method to include your own preprocessing steps. Passes
        through the input as is, by default.

        Args:
            x (T_Input | t.List[T_Input]): The input into the pipeline.

        Returns:
            T_Input | t.List[T_Input]: The processed input `x`, ready to pass through the
                embedding pipeline.
        """
        return x

    def _postprocess(
        self,
        y: np.ndarray | torch.Tensor | t.List[float] | t.List[t.List[float]],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> np.ndarray:
        """
        Performs postprocessing on the output of the model to convert it into
        a suitable `np.ndarray` ready to be used further.

        Override this method to include your own postprocessing steps. Passes
        through the input by converting to an `np.ndarray` if not already one,
        by default.

        Args:
            y (np.ndarray | torch.Tensor | t.List[float]):
                The output from the model.

        Returns:
            np.ndarray: The processed `np.ndarray`.
        """
        if isinstance(y, np.ndarray):
            return y
        return np.array(y)

    @abstractmethod
    def _run_model(
        self, x: T_Input | t.List[T_Input], *args: t.Any, **kwargs: t.Any
    ) -> np.ndarray | torch.Tensor | t.List[float] | t.List[t.List[float]]:
        """
        Runs the model on the input `x`.

        Args:
            x (T_Input | t.List[T_Input]): The input(s) to the model.

        Returns:
            (np.ndarray | torch.Tensor | t.List[float] | t.List[t.List[float]]):
                The output from the model.
        """
        pass

    def __call__(self, x: T_Input | t.List[T_Input]) -> np.ndarray:
        """
        Runs the EmbeddingPipeline on the input and returns the
        embedding as an `np.ndarray`. It includes three steps:
        1. Preprocessing: Call `self._preprocess` on the input `x`.
        2. Model Inference: Call `self._run_model` to perform inference on the
            preprocessed input.
        3. Postprocessing: Call `self._postprocess` to convert the model's
            output to a suitable `np.ndarray`.

        Args:
            x (T_Input | List[T_Input]): The input(s) into the pipeline.

        Returns:
            np.ndarray: The output embedding from the model.
        """
        x: T_Input | t.List[T_Input] = self._preprocess(x=x)
        y = self._run_model(x=x)
        embedding: np.ndarray = self._postprocess(y=y)
        return embedding
