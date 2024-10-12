import typing as t
import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
from clipseg import ClipSegProcessor
from pooler import Pooler
from segment import Segmenter

# ClipSegProcessor.run_model = timeit(ClipSegProcessor.run_model)

def main(img_dir: str, mask_dir) -> None:
    dataset = SegmentationDataset(img_dir, mask_dir,
                                #   transform=ToTensor(),
                                #   target_transform=ToTensor()
                                )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = ClipSegProcessor(["traversable area", "grass"])
    pooler = Pooler('max')
    segmenter = Segmenter('otsu')

    for i, (image, mask, img_path, mask_path) in enumerate(dataloader):
        image = image.squeeze(0).permute(1, 2, 0).numpy()
        mask = mask.squeeze(0).permute(1, 2, 0).numpy()
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        
        maps: t.List[np.ndarray] = model.run_model(image)

        pooled_map = pooler.pool(np.array(maps))
        segmented_map = segmenter.segment(pooled_map)

        image = cv2.resize(image, (image.shape[1]//(len(maps)+2), image.shape[0]//(len(maps)+2)))
        mask = cv2.resize(mask, (mask.shape[1]//(len(maps)+2), mask.shape[0]//(len(maps)+2)))
        superimposed = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        # print(f"image shape: {image.shape}, mask shape: {mask.shape}, pooled_map shape: {pooled_map.shape}, segmented_map shape: {segmented_map.shape}")
        superimposed_map = cv2.addWeighted(image, 0.7, segmented_map, 0.3, 0)

        
        output = [superimposed, superimposed_map] + maps

        output = cv2.hconcat(output)

        cv2.imshow("output", output)
        cv2.waitKey(10)
    cv2.destroyAllWindows()        
    
        

if __name__ == "__main__":
    main('../data/examples/offroad/rellis/combined_images', '../data/examples/offroad/rellis/combined_masks')
    

