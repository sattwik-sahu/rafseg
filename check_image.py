from PIL import Image

def check_image_channels(file_path):
    with Image.open(file_path) as img:
        if img.mode == 'RGB':
            return "RGB (3-dimensional)"
        elif img.mode in ['L', '1']:
            return "Grayscale/Binary (1-dimensional)"
        else:
            return f"Other ({img.mode})"

# Example usage
images = ['data/examples/offroad/deepscene/masks/b1-99445_mask.png',
          'data/examples/offroad/rellis/combined_masks/0001.png',
          'data/examples/offroad/rugd_masks_all/0001.png',
          'data/examples/offroad/yamaha/binary_masks/iid000000.png'
          ]

for file_path in images:
    result = check_image_channels(file_path)
    print(f"The image {file_path} is: {result}")