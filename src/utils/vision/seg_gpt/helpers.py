import numpy as np
from PIL.Image import Image
import torch
from utils.vision.seg_gpt.model import SegGPT
import typing as t
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Define constants
RES = HRES = 448
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def imagenet_normalize(x: np.ndarray) -> np.ndarray:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def convert_image_to_array(image: Image, resample: int | None = None) -> np.ndarray:
    return (
        np.array(image.convert(mode="RGB").resize(size=(RES, HRES), resample=resample))
        / 255.0
    )


def load_model(
    weights_path: str | Path,
    model_builder: t.Callable[[], SegGPT],
    seg_type: t.Literal["instance"] | str,
    device: t.Literal["cuda", "cpu"],
) -> SegGPT:
    # Build model
    model: SegGPT = model_builder()
    model.seg_type = seg_type

    # Load model params
    checkpoint = torch.load(f=weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=checkpoint["model"], strict=False)
    model.eval()
    return model


def plot_query_image_and_output_mask(
    query_image: Image, output_mask: torch.Tensor, title: str
) -> t.Tuple[Figure, t.List[Axes]]:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(nrows=1, ncols=2)
    query_image = np.array(query_image)

    # Plot original query image
    ax[0].imshow(query_image)
    ax[0].set_title("Query Image")

    # Plot query image with mask overlay
    ax[1].imshow(query_image)
    ax[1].imshow(output_mask[0], alpha=0.3)
    ax[1].set_title("Output")

    plt.title(title)

    return fig, ax


def plot_query_pipeline_prompts_and_output(
    prompt_images: t.List[Image],
    query_image: Image,
    output_mask: torch.Tensor,
    title: str,
):
    n_prompts = len(prompt_images)

    # Create a figure with a 2-column layout
    fig = plt.figure(figsize=(20, 5 * max(n_prompts, 2)))
    gs = fig.add_gridspec(max(n_prompts, 2), 2, width_ratios=[1, 2])

    # Display prompt images on the left
    for i, img in enumerate(prompt_images):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img)
        ax.set_title(f"Prompt {i+1}")
        ax.axis("off")

    # Create a sub-gridspec for the right column
    right_gs = gs[:, 1].subgridspec(2, 1, height_ratios=[1, 1])

    # Display query image on top right
    ax_query = fig.add_subplot(right_gs[0])
    ax_query.imshow(query_image)
    ax_query.set_title("Query Image", fontsize=14)
    ax_query.axis("off")

    # Display output (query image with mask overlay) on bottom right
    ax_output = fig.add_subplot(right_gs[1])
    ax_output.imshow(query_image)
    ax_output.imshow(
        output_mask[0], alpha=0.3, cmap="jet"
    )  # Using 'jet' colormap for better visibility
    ax_output.set_title("Output", fontsize=14)
    ax_output.axis("off")

    # Add title to the entire figure
    fig.suptitle(title, fontsize=20, y=0.98)  # Increased font size to 20

    # Adjust the layout to add space at the top
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # This leaves space at the top for the title

    return fig
