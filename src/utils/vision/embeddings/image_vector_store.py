from utils.vision.embeddings.vector_store import VectorStore
import numpy as np
import torch
from typing_extensions import override
from utils.vision.embeddings.create_store import PromptImageDocument
from pathlib import Path
from utils.vision.embeddings.base import EmbeddingPipeline
import typing as t
from PIL.Image import Image, open as open_image
from natsort import natsort
from glob import glob
import os
import pickle


class ImageVectorStore(VectorStore[PromptImageDocument]):
    def __init__(self) -> None:
        pass

    @override
    def _get_similarity_scores(
        self, query_vec: np.ndarray
    ) -> np.ndarray | torch.Tensor:
        similarity = torch.nn.CosineSimilarity()
        return similarity(query_vec, self._documents)


def create_image_vector_store_from_dirs(
    images_dir: str | Path,
    masks_dir: str | Path,
    embedding_pipeline: EmbeddingPipeline[t.Any, Image],
) -> ImageVectorStore:
    """
    Creates an ImageVectorStore by reading possible prompt images from
    `images_dir` and corresponding segmentation masks from `masks_dir`
    directories respectively.

    Args:
        images_dir (str | Path): The path to the directory containing all
            possible prompt images.
        masks_dir (str | Path): The path to the directory containing the
            corresponding segmentation masks for the images in `images_dir`.
        embedding_pipeline (EmbeddingPipeline[Any, Image]): The embedding
            pipeline for creating embeddings from the prompt images. It should
            take in `PIL.Image.Image` objects as input.

    Returns:
        ImageVectorStore: The `ImageVectorStore` object produced by storing the
            image path, mask path, image, mask, and embedding for the image in
            a document. The `ImageVectorStore` contains a list of these
            documents in the `self._documents` attribute, which may be accessed
            by using `.documents`.

    Note:
        `images_dir` and `masks_dir` should have the filenames same, or the order
        of files obtained after sorting the files from each of these folders must
        be the same.

        e.g. if `my_image.jpg` is the k-th image in `images_dir`, then the file
        in `masks_dir` at k-th position will be considered as its mask.
    """
    # Get all jpg, png images from the images_dir directory
    image_paths: t.List[Path] = natsort.natsorted(
        glob(f"{images_dir}/*.jpg") + glob(f"{images_dir}/*.png")
    )

    # Get all masks from the masks_dir directory (masks are stored only as png)
    mask_paths: t.List[Path] = natsort.natsorted(glob(os.path.join(masks_dir, "*.png")))

    # Get all images
    images: t.List[Image] = [open_image(image_path) for image_path in image_paths]

    # Embed all images in one go
    image_embeddings: np.ndarray = embedding_pipeline(x=images)

    # Initialize the vector store
    vector_store = ImageVectorStore()
    for i, (image_path, mask_path, image, embedding) in enumerate(
        zip(image_paths, mask_paths, images, image_embeddings)
    ):
        vector_store.add(
            doc=PromptImageDocument(
                id=i,
                embedding=embedding,
                image=image,
                mask=open_image(mask_path),
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    return vector_store


def load_image_vector_store(path: str | Path) -> ImageVectorStore:
    with open(path, "rb") as file:
        image_vector_store: ImageVectorStore = pickle.load(file=file)
    return image_vector_store
