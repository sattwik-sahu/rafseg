import typing as t
from pathlib import Path

import typer
from imgbeddings import imgbeddings as Imgbeddings
from rich.console import Console
from rich.progress import Progress
from typing_extensions import Annotated

from utils.vision.embeddings.image_vector_store import (
    ImageVectorStore,
    ingest_dir_to_image_vector_store,
)
from utils.vision.embeddings.imgbeddings_pipeline import ImgbeddingsPipeline
from utils.vision.embeddings.dino_pipeline import DinoPipeline
from utils.vision.embeddings.vit_pipeline import VitPipeline

app = typer.Typer(name="index", help="Create a vector store")
console = Console()


@app.command(
    name="images",
    help="Index images and masks from directories to an image vector store.",
)
def command(
    image_dir: Annotated[t.List[Path], typer.Option(help="The dir containing images")],
    mask_dir: Annotated[t.List[Path], typer.Option(help="The dir containing masks")],
    embedding_model: Annotated[
        str, typer.Argument(help='Emedding Model to Use: "clip" | "dino | vit"')
    ],
    vector_store_path: Annotated[
        Path, typer.Argument(help="The path to save the image vector store")
    ],
):
    # Create the embedding pipeline
    if embedding_model == "clip":
        embedding_pipeline = ImgbeddingsPipeline(model=Imgbeddings(gpu=True))
    elif embedding_model == "dino":
        embedding_pipeline = DinoPipeline()
    elif embedding_model == "vit":
        embedding_pipeline = VitPipeline()
    else:
        raise Exception("Please choose embedding model correctly [dino, clip, vit]")

    # Create the image vector store
    image_vector_store = ImageVectorStore()

    # Start ingesting image, mask dir pairs
    assert len(image_dir) == len(mask_dir)
    n_dirs = len(image_dir)

    with Progress() as progress:
        root_task = progress.add_task(
            description="Ingest folders to image vector store", total=n_dirs
        )
        for i, (image_dir_, mask_dir_) in enumerate(zip(image_dir, mask_dir)):
            console.log(
                f"Ingesting dir {i + 1}: {image_dir_.as_posix()} | {mask_dir_.as_posix()}"
            )
            ingest_dir_to_image_vector_store(
                images_dir=image_dir_,
                masks_dir=mask_dir_,
                embedding_pipeline=embedding_pipeline,
                image_vector_store=image_vector_store,
                store_images=False,
            )
            console.log(f"Stored {len(image_vector_store.documents)} documents.")
            progress.advance(root_task)
    console.log(
        f"Created image vector store with {len(image_vector_store.documents)} documents"
    )

    with console.status(f"Saving image vector store to {vector_store_path.as_posix()}"):
        image_vector_store.save(path=vector_store_path)


def main():
    app()


if __name__ == "__main__":
    main()
