from pathlib import Path

import numpy as np
import torch
import typer
from matplotlib import pyplot as plt
from PIL.Image import open as open_image
from rich.console import Console
from rich.progress import Progress
from typing_extensions import Annotated

import random

from ragseg.pipeline import Pipeline
from assisted_pilot.iou import calculate_binary_iou
from utils.vision.seg_gpt.helpers import plot_query_pipeline_prompts_and_output

app = typer.Typer()
console = Console()


@app.command()
def main(
    images: Annotated[Path, typer.Argument(help="Test images directory")],
    masks: Annotated[Path, typer.Argument(help="Test masks directory")],
    embedding_model: Annotated[str, typer.Option(help="Embedding model to use: dino | clip")],
    vector_store_path: Annotated[Path, typer.Option(help="Path to the vector store")],
    dev: Annotated[
        bool,
        typer.Option(
            help="Whether in dev mode. All segmentation output plots are shown one-by-one in this mode"
        ),
    ] = True,
    seg_gpt_weights_path: Annotated[
        Path, typer.Option(help="Path to the SegGPT weights")
    ] = Path("data/weights/seggpt_vit_large.pth"),
):
    # Create pipeline
    with console.status("Creating pipeline..."):
        pipeline = Pipeline(
            vector_store_path=vector_store_path,
            seg_gpt_weights_path=seg_gpt_weights_path,
            embedding_model=embedding_model,
        )

    # Image paths
    test_images_dir = images
    test_masks_dir = masks

    # Set top-k parameter
    k: int = 3

    # Get all common image filenames from the folders
    test_image_paths = sorted(
        list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")),
        key=lambda x: x.name,
    )
    test_mask_paths = sorted(list(test_masks_dir.glob("*.png")), key=lambda x: x.name)

    console.print(test_image_paths, test_mask_paths)
    image_mask_combined = list(zip(test_image_paths, test_mask_paths))
    random.shuffle(image_mask_combined)
    # Run pipeline on all images
    mean_iou: float = 0.0
    with Progress() as progress:
        task = progress.add_task("Evaluating images", total=len(test_image_paths))
        for i, (image_path, mask_path) in enumerate(image_mask_combined):

            console.log({"query_image_path": image_path, "query_mask_path": mask_path})

            query_image = open_image(image_path)
            best_matches, output = pipeline.run(x=query_image, k=k)

            console.log({
                "prompt_image_paths": [doc.image_path for doc in best_matches],
                "prompt_mask_paths": [doc.mask_path for doc in best_matches],
                "match_scores": "not available yet"
            })

            output_mask: torch.Tensor = 255.0 * (output[0] >= 128)
            test_image_mask = torch.tensor(np.array(open_image(mask_path)))

            # if dev:
            #     plot_query_pipeline_prompts_and_output(
            #         prompt_images=[
            #             m.image or np.array(open_image(m.image_path))
            #             for m in best_matches
            #         ],
            #         prompt_masks=[
            #             m.mask or np.array(open_image(m.mask_path))
            #             for m in best_matches
            #         ],
            #         query_image=query_image,
            #         query_ground_truth=test_image_mask,
            #         output_mask=output_mask,
            #         title="Testing pipeline with Prompts",
            #     )
            #     plt.show()

            # Calculate iou and update mean iou
            iou: float = calculate_binary_iou(pred=output_mask, target=test_image_mask)
            if iou < 0.5 and dev:
                plot_query_pipeline_prompts_and_output(
                    prompt_images=[
                        m.image or np.array(open_image(m.image_path))
                        for m in best_matches
                    ],
                    prompt_masks=[
                        m.mask or np.array(open_image(m.mask_path))
                        for m in best_matches
                    ],
                    query_image=query_image,
                    query_ground_truth=test_image_mask,
                    output_mask=output_mask,
                    title="Testing pipeline with Prompts",
                )
                plt.show()
            if iou == 0.0:
                iou = mean_iou
            mean_iou += (iou - mean_iou) / (i + 1)
            console.log(
                f"[yellow][{i + 1}][/] [dim]IoU = {iou}[/] | MIoU = [bright_cyan]{mean_iou}[/]"
            )
            progress.advance(task)


if __name__ == "__main__":
    app()
