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
    prompt_images: t.List[Image | np.ndarray],
    prompt_masks: t.List[Image | np.ndarray],
    query_image: Image,
    output_mask: torch.Tensor,
    title: str,
    query_ground_truth: t.Optional[Image | torch.Tensor] = None,
):
    n_prompts = len(prompt_images)

    # Determine the number of rows for the right column
    right_rows = 3 if query_ground_truth is not None else 2

    # Create a figure with a 3-column layout (prompt images, prompt masks, query/output)
    fig = plt.figure(figsize=(30, 5 * max(n_prompts, right_rows)))
    gs = fig.add_gridspec(
        max(n_prompts, right_rows), 3, width_ratios=[1, 1, 2], hspace=0.4, wspace=0.3
    )

    # Function to display an image with a title
    def display_image(ax, img, title, is_mask=False):
        if is_mask:
            ax.imshow(img, cmap="jet", alpha=0.7)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=16, pad=10)
        ax.axis("off")

    # Display prompt images and masks on the left two columns
    for i, (img, mask) in enumerate(zip(prompt_images, prompt_masks)):
        # Prompt image
        ax_img = fig.add_subplot(gs[i, 0])
        display_image(ax_img, img, f"Prompt {i+1}")

        # Prompt mask
        ax_mask = fig.add_subplot(gs[i, 1])
        display_image(ax_mask, mask, f"Prompt {i+1} Mask", is_mask=True)

    # Create a sub-gridspec for the right column
    right_gs = gs[:, 2].subgridspec(
        right_rows, 1, height_ratios=[1] * right_rows, hspace=0.3
    )

    # Display query image on top right
    ax_query = fig.add_subplot(right_gs[0])
    display_image(ax_query, query_image, "Query Image")

    # Display query ground truth if provided
    if query_ground_truth is not None:
        ax_ground_truth = fig.add_subplot(right_gs[1])
        display_image(ax_ground_truth, query_ground_truth, "Query Ground Truth")

    # Display output (query image with mask overlay) on bottom right
    ax_output = fig.add_subplot(right_gs[-1])
    ax_output.imshow(query_image)
    ax_output.imshow(output_mask, alpha=0.3, cmap="jet")
    ax_output.set_title("Output", fontsize=16, pad=10)
    ax_output.axis("off")

    # Add title to the entire figure
    fig.suptitle(title, fontsize=20, y=1.02)

    # Adjust the layout
    plt.tight_layout()

    # Check if the figure is too large and reduce size if necessary
    fig_size = fig.get_size_inches()
    if fig_size[1] > 30:  # If height is greater than 30 inches
        scale_factor = 30 / fig_size[1]
        new_size = (fig_size[0] * scale_factor, 30)
        fig.set_size_inches(new_size)
        plt.tight_layout()  # Re-adjust layout after resizing

    return fig
