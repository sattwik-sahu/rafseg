import os
import cv2
import argparse
from natsort import natsorted
import typing as t
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.table import Table


def display_video_parameters(
    console: Console,
    folder_path: str,
    output_path: str,
    frame_rate: int,
    image_count: int,
) -> None:
    """
    Display the video parameters in a formatted table.

    Args:
        console (Console): Rich console object for output.
        folder_path (str): Path to the folder containing images.
        output_path (str): Path for the output video file.
        frame_rate (int): Frame rate of the output video.
        image_count (int): Number of images to be processed.
    """
    table = Table(title="Video Conversion Parameters")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Input Folder", folder_path)
    table.add_row("Output File", output_path)
    table.add_row("Frame Rate", str(frame_rate))
    table.add_row("Number of Images", str(image_count))
    table.add_row("Estimated Duration", f"{image_count/frame_rate:.2f} seconds")

    console.print(table)


def create_video_from_images(
    folder_path: str, output_path: str, frame_rate: int
) -> None:
    """
    Create a video from a series of images in a specified folder.

    This function reads all image files from a given folder, sorts them naturally,
    and creates a video using these images as frames. The resulting video is saved
    in MP4 format. It displays the video parameters before starting and shows a
    progress bar during the conversion process.

    Args:
        folder_path (str): Path to the folder containing the image files.
        output_path (str): Path where the output video file will be saved (including .mp4 extension).
        frame_rate (int): Frame rate (FPS) for the output video.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified folder_path does not exist.
        ValueError: If no valid image files are found in the specified folder.
        RuntimeError: If there's an error during video creation.

    Note:
        Supported image formats are: .png, .jpg, .jpeg, .tiff, .bmp, and .gif.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder does not exist: {folder_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get list of image files in the folder
    image_files: t.List[str] = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]

    # Sort the files naturally (1, 2, 3, 10, 11, 12 instead of 1, 10, 11, 12, 2, 3)
    image_files = natsorted(image_files)

    if not image_files:
        raise ValueError(f"No valid image files found in {folder_path}")

    console = Console()

    # Display video parameters
    display_video_parameters(
        console, folder_path, output_path, frame_rate, len(image_files)
    )

    # Read the first image to get dimensions
    first_image: t.Any = cv2.imread(os.path.join(folder_path, image_files[0]))
    if first_image is None:
        raise RuntimeError(f"Failed to read the first image: {image_files[0]}")

    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
    out: cv2.VideoWriter = cv2.VideoWriter(
        output_path, fourcc, frame_rate, (width, height)
    )

    if not out.isOpened():
        raise RuntimeError(
            f"Failed to create video writer. Check if the output path is valid: {output_path}"
        )

    # Create a progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Adding frames to video...", total=len(image_files)
        )

        # Iterate through images and add to video
        for image in image_files:
            image_path: str = os.path.join(folder_path, image)
            frame: t.Any = cv2.imread(image_path)
            if frame is None:
                console.print(
                    f"[yellow]Warning:[/yellow] Failed to read image: {image_path}"
                )
                continue
            out.write(frame)
            progress.update(task, advance=1)

    # Release the VideoWriter
    out.release()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        console.print(f"[green]Video created successfully:[/green] {output_path}")
    else:
        raise RuntimeError(
            f"Failed to create video. The output file is missing or empty: {output_path}"
        )


def main() -> None:
    """
    Main function to parse command line arguments and call the video creation function.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create a video from images in a folder."
    )
    parser.add_argument("folder_path", help="Path to the folder containing images")
    parser.add_argument(
        "-o",
        "--output",
        help="Path for the output video file (including .mp4 extension)",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=30,
        help="Frame rate of the output video (default: 30)",
    )

    args: argparse.Namespace = parser.parse_args()

    # If output path is not specified, create one based on the input folder
    if args.output is None:
        folder_name = os.path.basename(os.path.normpath(args.folder_path))
        args.output = os.path.join(args.folder_path, f"{folder_name}_output.mp4")

    try:
        create_video_from_images(args.folder_path, args.output, args.frame_rate)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        Console().print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
