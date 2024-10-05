import cv2
import matplotlib.pyplot as plt
import mplcursors
import os

image_dir = 'data/examples/offroad/deepscene/images'
mask_dir = 'data/examples/offroad/deepscene/masks_multi_class'
image_paths = [image_dir + '/' +  x for x in os.listdir(image_dir)]
mask_paths = [mask_dir + '/' + x for x in os.listdir(mask_dir)]
image_paths.sort()
mask_paths.sort()

def show_image_and_mask(index: int):
    # Read the image and mask
    image = cv2.imread(image_paths[index])
    mask = cv2.imread(mask_paths[index], cv2.IMREAD_GRAYSCALE)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image and mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Image')
    axes[0].axis('off')

    mask_plot = axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    # Add colorbar
    cbar = plt.colorbar(mask_plot, ax=axes[1], orientation='vertical', label='Intensity')

    # Use mplcursors to show pixel values on hover
    cursor = mplcursors.cursor(mask_plot, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target.index
        value = mask[y, x]
        sel.annotation.set_text(f'Intensity: {value}')

    plt.tight_layout()
    plt.show()

# Assuming image_paths and mask_paths are defined
for i in range(len(mask_paths)):
    show_image_and_mask(i)