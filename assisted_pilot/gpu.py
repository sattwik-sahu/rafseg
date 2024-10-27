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
from map_diff import process_attention_maps, process_attention_maps_weighted

def main(img_dir: str, mask_dir) -> None:
    dataset = SegmentationDataset(img_dir, mask_dir)
    indices = list(range(0, len(dataset), 6))
    subset_dataset = Subset(dataset, indices)
    dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)
    
    positive_anchors = ["grass"]
    negative_anchors = ["large shrubs", "puddle"]
    anchors = {"positive": positive_anchors, "negative": negative_anchors}
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClipSegProcessor(anchors)
    
    pooler = Pooler('max')
    segmenter = Segmenter('otsu')
    miou = 0
    progress_bar = tqdm(dataloader, desc='Processing images')

    for i, (image, mask, img_path, mask_path) in enumerate(progress_bar):
        # Convert image to tensor and move to GPU
        image = image.squeeze(0).permute(1, 2, 0).numpy()
        image = cv2.resize(image, (1920, image.shape[0]*1920//image.shape[1]))
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
        
        mask = mask.squeeze(0).permute(1, 2, 0).numpy()
        mask = cv2.resize(mask, (1920, mask.shape[0]*1920//mask.shape[1]))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        print(image_tensor.device, model.device)
        
        # Run model on GPU
        maps: t.List[torch.Tensor] = model.run_model(image_tensor)
        
        # Move maps back to CPU for further processing
        maps = [m.cpu().numpy() for m in maps]

        positive_maps = maps[:len(model.prompts['positive'])]
        negative_maps = maps[len(model.prompts['positive']):]

        positive_pooled_map = pooler.pool(np.array(positive_maps))
        negative_pooled_map = pooler.pool(np.array(negative_maps))

        pooled_map = process_attention_maps_weighted(positive_pooled_map, negative_pooled_map, 1, 1)
        
        segmented_map = segmenter.segment(pooled_map)

        traversable, centre_cell_coords, cell_avg_value = bottom_square(pooled_map, 75)
        start_row, end_row, start_col, end_col = centre_cell_coords

        # Draw the bottom center cell on the segmented map
        pooled_map = cv2.rectangle(pooled_map, (start_col, start_row), (end_col, end_row), (0, 0, 255), 2)
        label_pooled = "traversable_pooled" if traversable else "non-traversable_pooled | Please Enter new prompt"
        pooled_map = cv2.putText(pooled_map, f"{label_pooled}", (start_col, start_row-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        segmented_map = cv2.rectangle(segmented_map, (start_col, start_row), (end_col, end_row), (0, 0, 255), 2)
        label = "traversable" if traversable else "non-traversable | Please Enter new prompt"
        segmented_map = cv2.putText(segmented_map, f"{label}", (start_col, start_row-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        image = cv2.resize(image, (image.shape[1]//(len(maps)+2), image.shape[0]//(len(maps)+2)))
        mask = cv2.resize(mask, (mask.shape[1]//(len(maps)+2), mask.shape[0]//(len(maps)+2)))
        superimposed = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        superimposed_map = cv2.addWeighted(image, 0.7, segmented_map, 0.3, 0)
        
        output = [superimposed, superimposed_map] + maps
        output = cv2.hconcat(output)

        iou = calculate_binary_iou(mask, segmented_map)
        miou += (iou - miou)/(i+1)
        progress_bar.set_postfix({'IOU': iou, 'mIOU': miou, 'cell Avg':cell_avg_value, 'Image': img_path})
        cv2.imshow(f"output", output)
        cv2.waitKey(10)

        if not(traversable):
            new_prompt = input("Enter new traversable terrain prompt: ")
            if new_prompt:
                model.prompts['positive'].append(new_prompt)

    cv2.destroyAllWindows()        

if __name__ == "__main__":
    main('../data/examples/offroad/rellis/combined_images', '../data/examples/offroad/rellis/combined_masks')