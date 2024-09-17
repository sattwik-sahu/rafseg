import shutil
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import track

# Initialize Typer and Rich Console
app = typer.Typer()
console = Console()

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}


def copy_images(input_dir: Path, output_dir: Path):
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of image files to copy
    image_files = [
        f for f in input_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Copy images with progress bar
    for image in track(image_files, description="Copying images..."):
        try:
            shutil.copy(image, output_dir)
        except Exception as e:
            console.print(f"[bold red]Error copying {image}: {e}[/bold red]")

    console.print(
        f"[bold green]Finished copying {len(image_files)} images to {output_dir}[/bold green]"
    )


@app.command()
def main(input_dir: str, output_dir: str):
    """Recursively copy all images from input_dir to output_dir."""
    # Convert the string paths to Path objects
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    # Check if input directory exists
    if not input_path.is_dir():
        console.print(
            f"[bold red]Input directory '{input_path}' does not exist![/bold red]"
        )
        raise typer.Exit(code=1)

    # Call the copy function
    copy_images(input_path, output_path)


if __name__ == "__main__":
    app()
