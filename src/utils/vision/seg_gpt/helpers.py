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
