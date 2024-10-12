from transformers import AutoProcessor, CLIPSegForImageSegmentation
import typing as t
import torch
import numpy as np
import cv2

class ClipSegProcessor():
    def __init__(self, initial_prompts: t.List[str]) -> None:
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.prompts = initial_prompts

    def post_process_outputs(self, outputs, image_shape, image_dtype) -> t.List[np.ndarray]:
        """
        Post process model outputs into a list of attention maps: [np.ndarray]
        """
        maps = []
        
        preds = outputs.logits.unsqueeze(1)
        for pred in preds:
            sigmoided_pred = torch.sigmoid(pred[0]).cpu().numpy()*255
            resized = cv2.resize(sigmoided_pred, (image_shape[1]//(len(preds)+2), image_shape[0]//(len(preds)+2)))
            rbg = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            maps.append(rbg.astype(image_dtype))

        # maps: t.List[np.ndarray] = [torch.sigmoid(pred[0]).cpu().numpy()*255 for pred in preds]
        return maps


    # @timeit
    def run_model(self, image) -> t.List[np.ndarray]:
        """
        Return a list of attention maps. Length of list = number of prompt texts.
        """
        inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        maps: t.List[np.ndarray] = self.post_process_outputs(outputs, image.shape, image.dtype)
        return maps
        



