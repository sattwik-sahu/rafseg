import typing as t
from pathlib import Path
from time import time_ns

import numpy as np
from matplotlib import pyplot as plt
from PIL.Image import open as open_image
from torchvision.transforms.functional import InterpolationMode, resize

from utils.vision.seg_gpt import SegGPT_Inference
from utils.vision.seg_gpt.helpers import load_model
from utils.vision.seg_gpt.model import (
    model_builder as model_builder,
)


def main():
    # Create the model
    model = load_model(
        weights_path="data/weights/seggpt_vit_large.pth",
        model_builder=model_builder,
        device="cuda",
        seg_type="instance",
    ).to("cuda")

    def _example_path(path: str) -> Path:
        examples_root: str = "data/examples/offroad"
        return Path(examples_root, path)

    query_inx: int = 1
    prompt_inx: t.List[int] = [1]
    # query_image_path: str = _example_path(
    #     f"jungle__image_{f'{query_inx}'.zfill(3)}.jpg"
    # )
    query_image_path: Path = _example_path("rugd-sample/images/trail-11_00001.png")
    prompt_image_paths: t.List[Path] = [
        _example_path(f"jungle__image_{f'{i}'.zfill(3)}.jpg") for i in prompt_inx
    ]
    prompt_mask_paths: t.List[Path] = [
        _example_path(f"jungle__mask_{f'{i}'.zfill(3)}.png") for i in prompt_inx
    ]

    inference: SegGPT_Inference = SegGPT_Inference(model=model, device="cuda")
    t0 = time_ns()
    output = inference(
        query_image=open_image(query_image_path),
        prompt_images=[open_image(path) for path in prompt_image_paths],
        prompt_masks=[open_image(path) for path in prompt_mask_paths],
    )
    delta = (time_ns() - t0) * 1e-6
    print(f"Output size: {output.shape}")
    print(f"Inference took {delta} ms")

    # n_plots = output.shape[-1] + 1
    _, ax = plt.subplots(nrows=1, ncols=2)

    # for i, ax_ in enumerate(ax[:-1]):
    #     ax_.imshow(output[:, :, i] / 255.0)
    query_image = open_image(query_image_path)
    # output_ = resize(
    #     img=output, interpolation=InterpolationMode.NEAREST, size=query_image.size[::-1]
    # )
    output_ = output
    print(f"Query image size: {query_image.size}")
    print(f"Output size: {output_.size()}")
    ax[0].imshow(output_[0], cmap="inferno")
    ax[1].imshow(np.array(query_image))
    ax[1].imshow(output_[0], cmap="inferno", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
