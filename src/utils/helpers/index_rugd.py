from utils.vision.embeddings.image_vector_store import (
    ImageVectorStore,
    ingest_dir_to_image_vector_store,
)
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, TaskID
import typing as t
from utils.vision.embeddings.imgbeddings_pipeline import (
    ImgbeddingsPipeline,
    Imgbeddings,
)
import numpy as np


def main():
    console = Console()

    # The dataset root
    data_root = Path("data/examples/offroad/")

    # RUGD images and masks paths
    rugd_images_root = data_root.joinpath("rugd-full")
    rugd_masks_root = data_root.joinpath("rugd-masks")

    # RUGD images and masks folders
    rugd_image_paths: t.List[Path] = [Path(path) for path in rugd_images_root.glob("*")]
    rugd_mask_paths: t.List[Path] = [Path(path) for path in rugd_masks_root.glob("*")]

    console.log(
        f"Found {len(rugd_image_paths)} image folders, {len(rugd_mask_paths)} mask folders"
    )

    assert len(rugd_image_paths) == len(rugd_mask_paths)

    rand_path_inx = np.random.choice(len(rugd_image_paths), len(rugd_image_paths))
    inx, not_inx = rand_path_inx[:14], rand_path_inx[14:] 
    index_paths = {
        "image": [rugd_image_paths[i] for i in inx],
        "mask": [rugd_mask_paths[i] for i in inx],
    }
    console.log(f"Paths being indexed: {index_paths}")
    console.log(f"Paths not being indexed: {[rugd_image_paths[i] for i in not_inx]}")

    # Initialize image vector store
    image_vector_store = ImageVectorStore()

    # Create embedding pipeline
    embedding_pipeline = ImgbeddingsPipeline(model=Imgbeddings(gpu=True))

    with Progress(console=console) as progress:
        indexer_task: TaskID = progress.add_task(
            "Indexing RUGD dataset", total=len(rugd_image_paths)
        )
        for image_path, mask_path in zip(index_paths["image"], index_paths["mask"]):
            console.log(f"Ingesting episode {image_path.name}")
            ingest_dir_to_image_vector_store(
                images_dir=image_path,
                masks_dir=mask_path,
                embedding_pipeline=embedding_pipeline,
                image_vector_store=image_vector_store,
            )
            console.log(f"Ingested episode [cyan]{image_path.name}[/]")
        progress.advance(indexer_task)

    console.log(
        f"Created image vector store with {len(image_vector_store.documents)} documents"
    )

    store_filename = input("Vector store filename? ")

    with console.status("Saving vector store..."):
        image_vector_store.save(f"data/cache/{store_filename}.pkl")
    console.log("[light_green]Done[/]")


if __name__ == "__main__":
    main()
