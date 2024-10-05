import os
import shutil

def consolidate_images(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    counter = 1
    processed_files = set()

    for root, dirs, files in os.walk(source_dir):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                src_path = os.path.join(root, file)
                
                # Generate a unique identifier for each file
                relative_path = os.path.relpath(src_path, source_dir)
                file_id = f"{relative_path}_{file}"
                
                if file_id not in processed_files:
                    new_filename = f"{counter:04d}{os.path.splitext(file)[1]}"
                    dst_path = os.path.join(destination_dir, new_filename)
                    
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")
                    
                    processed_files.add(file_id)
                    counter += 1
                else:
                    print(f"Skipped duplicate file: {src_path}")

# Example usage
source_directory = "data/examples/offroad/rugd-masks"
destination_directory = "data/examples/offroad/rugd_masks_all"

consolidate_images(source_directory, destination_directory)