import os
from PIL import Image

def convert_binary_to_rgb(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Open the binary mask
            with Image.open(os.path.join(input_dir, filename)) as img:
                # Convert to RGB mode
                rgb_img = img.convert('RGB')

                # Ensure the image is black and white
                pixels = rgb_img.load()
                width, height = rgb_img.size
                for x in range(width):
                    for y in range(height):
                        r, g, b = pixels[x, y]
                        if r > 128 or g > 128 or b > 128:
                            pixels[x, y] = (255, 255, 255)
                        else:
                            pixels[x, y] = (0, 0, 0)

                # Save the new image
                output_filename = os.path.splitext(filename)[0] + '_rgb.png'
                rgb_img.save(os.path.join(output_dir, output_filename))

    print(f"Conversion complete. RGB images saved in {output_dir}")

# Usage example:
input_directory = 'data/examples/offroad/deepscene/masks'
output_directory = 'data/examples/offroad/deepscene/masks_rgb'
convert_binary_to_rgb(input_directory, output_directory)