import cv2
import matplotlib.pyplot as plt
import mplcursors
import os
import random

image_dir = "data/examples/offroad/rugd_images_all"
mask_dir = "data/examples/offroad/rugd_masks_all"

image_paths = [image_dir + '/' +  x for x in os.listdir(image_dir)]
mask_paths = [mask_dir + '/' + x for x in os.listdir(mask_dir)]
image_paths.sort()
mask_paths.sort()

def show_image_and_mask(index: int):
    # Read the image and mask
    image = cv2.imread(image_paths[index])
    mask = cv2.imread(mask_paths[index])

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Superimpose the image and mask with alpha=0.5
    superimposed = cv2.addWeighted(image_rgb, 0.5, mask_rgb, 0.5, 0)

    # Plot the superimposed image
    plt.imshow(superimposed)
    plt.title(f'Superimposed Image and Mask {image_paths[index]}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Assuming image_paths and mask_paths are defined
for i in range(len(mask_paths)):
    index = random.randint(0, len(mask_paths)-1)
    show_image_and_mask(index)