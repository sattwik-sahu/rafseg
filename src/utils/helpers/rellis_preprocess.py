import os
from pathlib import Path
import numpy as np
from PIL import Image
import typer
from rich.console import Console
from rich.progress import Progress, TaskID

# Initialize Typer and Rich Console
app = typer.Typer()
console = Console()

# Define the terrain classes
terrain_classes = {
    "traversible": [1, 3, 10, 23, 31, 33],
    "nonTraversible": [0, 4, 5, 6, 7, 8, 9, 12, 15, 17, 18, 19, 27, 29, 30, 34],
}
# Convert to sets for faster lookup
traversible_set = set(terrain_classes["traversible"])
non_traversible_set = set(terrain_classes["nonTraversible"])


def process_image(image_path: Path, progress: Progress, task: TaskID):
    """Modify the image according to the terrain classes."""
    # Open the image as a numpy array
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)

    # Create an empty output array
    output_array = np.zeros_like(img_array, dtype=np.uint8)

    # Map traversible pixels to 255, and nonTraversible to 0
    output_array[np.isin(img_array, list(traversible_set))] = 255
    output_array[np.isin(img_array, list(non_traversible_set))] = 0

    # Convert the numpy array back to an image
    output_image = Image.fromarray(output_array)

    # Save the modified image, replacing the original
    output_image.save(image_path)
    progress.update(task, advance=1)


@app.command()
def main(masks_folder: str):
    """Modify all images in masks_folder to set traversible pixels to 255 and nonTraversible to 0."""
    masks_path = Path(masks_folder).resolve()

    # Ensure the folder exists
    if not masks_path.is_dir():
        console.print(
            f"[bold red]Error: The folder '{masks_path}' does not exist![/bold red]"
        )
        raise typer.Exit(code=1)

    # Get the list of PNG images in the folder
    image_paths = list(masks_path.glob("*.png"))
    total_images = len(image_paths)

    # Process each PNG image in the folder with a progress bar
    with Progress() as progress:
        task = progress.add_task("[green]Processing images...", total=total_images)
        for image_path in image_paths:
            process_image(image_path, progress, task)

    console.print(
        "[bold green]All images have been processed successfully![/bold green]"
    )


if __name__ == "__main__":
    app()
