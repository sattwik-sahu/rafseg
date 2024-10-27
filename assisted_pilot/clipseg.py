from transformers import AutoProcessor, CLIPSegForImageSegmentation
import typing as t
import torch
import numpy as np
import cv2

class ClipSegProcessor():
    def __init__(self, initial_prompts: t.Dict[str, t.List[str]]) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        allowed_keys = {"positive", "negative"}
        if set(initial_prompts.keys()) != allowed_keys:
            raise ValueError(f"initial_prompts must contain exactly the keys: {allowed_keys}")
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
        prompts = self.prompts['positive'] + self.prompts['negative']
        # print(prompts)
        inputs = self.processor(text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        maps = []
        for ix, map in enumerate(self.post_process_outputs(outputs, image.shape, image.dtype)):
            #write map prompt text on image
            map = cv2.putText(map, f"{prompts[ix]}", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            maps.append(map)
        return maps
        



