from utils.vision.embeddings.vector_store import VectorStore, DocumentVector
from PIL.Image import Image, open as open_image
import typing as t
from pathlib import Path
import numpy as np
from glob import glob
import os
from dataclasses import dataclass
from utils.vision.embeddings.base import EmbeddingPipeline
from utils.vision.embeddings.imgbeddings_pipeline import (
    ImgbeddingsPipeline,
    Imgbeddings,
)
from natsort import natsort
import pickle
from rich.console import Console


@dataclass
class PromptImageDocument(DocumentVector):
    image: Image | None
    mask: Image | None
    image_path: str | Path
    mask_path: str | Path


def create_image_vector_store_from_dirs(
    images_dir: str | Path,
    masks_dir: str | Path,
    embedding_pipeline: EmbeddingPipeline[t.Any, Image],
    store_pil_images: bool = True,
) -> VectorStore:
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
    vector_store = VectorStore()
    for i, (image_path, mask_path, image, embedding) in enumerate(
        zip(image_paths, mask_paths, images, image_embeddings)
    ):
        mask = open_image(mask_path)
        if not store_pil_images:
            image = None
            mask = None

        vector_store.add(
            doc=PromptImageDocument(
                id=i,
                embedding=embedding,
                image=image,
                mask=mask,
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return vector_store


def main():
    console = Console()

    images_dir = "data/examples/offroad/offterrain-attention-kaggle/images"
    masks_dir = "data/examples/offroad/offterrain-attention-kaggle/masks"
    vector_store_cache_path = "data/cache/vector_store.pkl"

    console.log("Initializing embedding pipeline")
    embedding_pipeline = ImgbeddingsPipeline(model=Imgbeddings(gpu=True))

    with console.status(
        f"Creating vector store from paths {images_dir} and {masks_dir}"
    ):
        vector_store = create_image_vector_store_from_dirs(
            images_dir=images_dir,
            masks_dir=masks_dir,
            embedding_pipeline=embedding_pipeline,
        )
    console.log(f"Created vector store with {len(vector_store.documents)} documents")

    # Save vector store to file
    with console.status(f"Saving vector store to {vector_store_cache_path}"):
        with open(vector_store_cache_path, "wb") as cache_file:
            pickle.dump(vector_store, file=cache_file)
    console.log("[lightgreen]Done.[/]")


if __name__ == "__main__":
    main()
