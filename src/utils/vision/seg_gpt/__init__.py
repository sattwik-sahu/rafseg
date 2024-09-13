import typing as t
from utils.vision.seg_gpt.model import SegGPT
from utils.vision.seg_gpt.helpers import imagenet_normalize, convert_image_to_array
from PIL.Image import Image
import numpy as np
import torch
from utils.vision.seg_gpt.engine import run_one_image
from torchvision.transforms.functional import resize, InterpolationMode


class SegGPT_Inference:
    def __init__(self, model: SegGPT, device: t.Literal["cpu", "cuda"]) -> None:
        self.model = model
        self.device = device

    def _preprocess_images(
        self,
        query_image: Image,
        prompt_images: t.List[Image],
        prompt_masks: t.List[Image],
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        query_image_arr: np.ndarray = convert_image_to_array(image=query_image)
        query_image_norm: np.ndarray = imagenet_normalize(query_image_arr)

        # Create prompt image and mask batches
        prompt_image_batch: np.ndarray = np.stack(
            [
                np.concatenate(
                    (
                        imagenet_normalize(convert_image_to_array(image=image)),
                        query_image_norm,
                    )
                )
                for image in prompt_images
            ],
            axis=0,
        )
        prompt_mask_batch: np.ndarray = np.stack(
            [
                np.concatenate(
                    [imagenet_normalize(convert_image_to_array(image=mask))] * 2
                )
                for mask in prompt_masks
            ],
            axis=0,
        )

        return prompt_image_batch, prompt_mask_batch
    
    def _postprocess(self, output: torch.Tensor, query_image: Image) -> torch.Tensor:
        query_image_size = query_image.size[::-1]   # PIL stores size in (W, H) format
        output = output.permute([2, 0, 1])
        return resize(
            img=output,
            interpolation=InterpolationMode.NEAREST,
            size=query_image_size
        )

    def __call__(
        self,
        query_image: Image,
        prompt_images: t.List[Image],
        prompt_masks: t.List[Image],
    ) -> torch.Tensor:
        # Prepare batch of prompt image and mask
        image_batch, target_batch = self._preprocess_images(
            query_image=query_image,
            prompt_images=prompt_images,
            prompt_masks=prompt_masks,
        )

        # Run inference on the preprocessed batches now
        output = run_one_image(
            img=image_batch,
            tgt=target_batch,
            model=self.model,
            device=self.device,
        )

        # Postprocess the output
        postprocessed_output: torch.Tensor = self._postprocess(output=output, query_image=query_image)

        return postprocessed_output
