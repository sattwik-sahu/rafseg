import os
import cv2
import numpy as np

def process_masks(input_dir, output_dir, traversable, non_traversable):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Read the image in grayscale
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Create binary mask
            mask = np.zeros_like(img)

            # Set traversable pixels to white (255)
            for value in traversable:
                mask[img == value] = 255

            # Set non-traversable pixels to black (0)
            for value in non_traversable:
                mask[img == value] = 0

            # Write the new image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, mask)

            print(f"Processed: {filename}")

    print("All masks have been processed.")

# Example usage
input_directory = "data/examples/offroad/deepscene/masks_multi_class"
output_directory = "data/examples/offroad/deepscene/masks"

# Define traversable and non-traversable values
traversable_values = [149, 170]
non_traversable_values = [0, 35, 96, 99]

process_masks(input_directory, output_directory, traversable_values, non_traversable_values)