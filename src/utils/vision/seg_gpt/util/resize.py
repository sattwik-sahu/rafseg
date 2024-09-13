import argparse
import os
from PIL import Image

def resize_images(input_dir, out_height, out_width):
    # Create the output directory
    output_dir = os.path.join(input_dir, "resized")
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(input_path) as img:
                # Resize the image
                resized_img = img.resize((out_width, out_height), Image.LANCZOS)
                # Save the resized image
                resized_img.save(output_path)
                print(f"Resized {filename} to {out_width}x{out_height}")

def main():
    parser = argparse.ArgumentParser(description="Resize images in a directory.")
    parser.add_argument("--input-dir", required=True, help="Input directory containing images")
    parser.add_argument("--out-height", type=int, required=True, help="Output height in pixels")
    parser.add_argument("--out-width", type=int, required=True, help="Output width in pixels")
    
    args = parser.parse_args()

    resize_images(args.input_dir, args.out_height, args.out_width)

if __name__ == "__main__":
    main()