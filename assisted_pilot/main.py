import typing as t
import torch
from torch.utils.data import DataLoader, Subset
from dataset import SegmentationDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
from clipseg import ClipSegProcessor
from pooler import Pooler
from segment import Segmenter
from iou import calculate_binary_iou
from tqdm import tqdm
from criteria import bottom_square



# ClipSegProcessor.run_model = timeit(ClipSegProcessor.run_model)

def main(img_dir: str, mask_dir) -> None:
    dataset = SegmentationDataset(img_dir, mask_dir,
                                #   transform=ToTensor(),
                                #   target_transform=ToTensor()
                                )
    # Get indices for every third image
    indices = list(range(0, len(dataset), 1))

    # Create subset dataset
    subset_dataset = Subset(dataset, indices)
    dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=True)
    model = ClipSegProcessor(["grass","path"])
    pooler = Pooler('max')
    segmenter = Segmenter('otsu')
    miou = 0
    progress_bar = tqdm(dataloader, desc='Processing images')

    for i, (image, mask, img_path, mask_path) in enumerate(progress_bar):
        image = image.squeeze(0).permute(1, 2, 0).numpy()
        image = cv2.resize(image, (1920, image.shape[0]*1920//image.shape[1]))
        mask = mask.squeeze(0).permute(1, 2, 0).numpy()
        mask = cv2.resize(mask, (1920, mask.shape[0]*1920//mask.shape[1]))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        
        maps: t.List[np.ndarray] = model.run_model(image)

        pooled_map = pooler.pool(np.array(maps))
        
        segmented_map = segmenter.segment(pooled_map)

        traversable, centre_cell_coords = bottom_square(segmented_map, 100)
        start_row, end_row, start_col, end_col = centre_cell_coords

        # Draw the bottom center cell on the segmented map
        pooled_map = cv2.rectangle(pooled_map, (start_col, start_row), (end_col, end_row), (0, 0, 255), 2)
        label = "traversable" if traversable else "non-traversable | Please Enter new prompt"
        #label rectangle
        pooled_map = cv2.putText(pooled_map, f"{label}", (start_col, start_row-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw the bottom center cell on the segmented map
        segmented_map = cv2.rectangle(segmented_map, (start_col, start_row), (end_col, end_row), (0, 0, 255), 2)
        label = "traversable" if traversable else "non-traversable | Please Enter new prompt"
        #label rectangle
        segmented_map = cv2.putText(segmented_map, f"{label}", (start_col, start_row-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



        image = cv2.resize(image, (image.shape[1]//(len(maps)+2), image.shape[0]//(len(maps)+2)))
        mask = cv2.resize(mask, (mask.shape[1]//(len(maps)+2), mask.shape[0]//(len(maps)+2)))
        superimposed = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        # print(f"image shape: {image.shape}, mask shape: {mask.shape}, pooled_map shape: {pooled_map.shape}, segmented_map shape: {segmented_map.shape}")
        superimposed_map = cv2.addWeighted(image, 0.7, pooled_map, 0.3, 0)

        
        output = [superimposed, superimposed_map] + maps

        

        output = cv2.hconcat(output)



        # output = [cv2.resize(img, (1080, image.shape[0]*1080//image.shape[1])) for img in output]
        iou = calculate_binary_iou(mask, segmented_map)
        miou += (iou - miou)/(i+1)
        # print(f"IOU: {iou}, mIOU: {miou}")
        progress_bar.set_postfix({'IOU': iou, 'mIOU': miou})
        cv2.imshow(f"output", output)
        cv2.waitKey(10)
        if not(traversable):
            new_prompt = input("Enter new traversable terrain prompt: ")
            model.prompts.append(new_prompt)
    cv2.destroyAllWindows()        
    
        

if __name__ == "__main__":
    main('../data/examples/offroad/rellis/combined_images', '../data/examples/offroad/rellis/combined_masks')
    

