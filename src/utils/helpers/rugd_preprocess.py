from PIL import Image
import numpy as np
import os
from rich.progress import Progress
from rich.console import Console
from enum import Enum, auto
import typing as t


class RUGDClass(Enum):
    VOID = ([0, 0, 0], auto())
    DIRT = ([108, 64, 20], auto())
    SAND = ([255, 229, 204], auto())
    GRASS = ([0, 102, 0], auto())
    TREE = ([0, 255, 0], auto())
    POLE = ([0, 153, 153], auto())
    WATER = ([0, 128, 255], auto())
    SKY = ([0, 0, 255], auto())
    VEHICLE = ([255, 255, 0], auto())
    CONTAINER = ([255, 0, 127], auto())
    ASPHALT = ([64, 64, 64], auto())
    GRAVEL = ([255, 128, 0], auto())
    BUILDING = ([255, 0, 0], auto())
    MULCH = ([153, 76, 0], auto())
    ROCK_BED = ([102, 102, 0], auto())
    LOG = ([102, 0, 0], auto())
    BICYCLE = ([0, 255, 128], auto())
    PERSON = ([204, 153, 255], auto())
    FENCE = ([102, 0, 204], auto())
    BUSH = ([255, 153, 204], auto())
    SIGN = ([0, 102, 102], auto())
    ROCK = ([153, 204, 255], auto())
    BRIDGE = ([102, 255, 255], auto())
    CONCRETE = ([101, 101, 11], auto())
    PICNIC_TABLE = ([114, 85, 47], auto())

    def __init__(self, rgb, _):
        self.rgb = rgb


def rgb_to_binary_mask(input_path: str, output_path: str, target_classes):
    with Image.open(input_path) as img:
        img_array = np.array(img)

    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

    for cls in target_classes:
        class_mask = np.all(img_array == cls.rgb, axis=-1)
        mask = np.logical_or(mask, class_mask)

    mask = mask.astype(np.uint8) * 255

    mask_img = Image.fromarray(mask)
    mask_img.save(output_path)


def process_rugd_dataset(
    input_rugd_folder: str, output_rugd_folder: str, target_classes: t.List[RUGDClass]
):
    console = Console()

    episode_folders = [
        f
        for f in os.listdir(input_rugd_folder)
        if os.path.isdir(os.path.join(input_rugd_folder, f))
    ]

    with Progress() as progress:
        main_task = progress.add_task(
            "[bold green]Processing RUGD dataset...", total=len(episode_folders)
        )

        for episode_folder in episode_folders:
            input_episode_path = os.path.join(input_rugd_folder, episode_folder)
            output_episode_path = os.path.join(output_rugd_folder, episode_folder)

            os.makedirs(output_episode_path, exist_ok=True)

            png_files = [
                f for f in os.listdir(input_episode_path) if f.endswith(".png")
            ]

            episode_task = progress.add_task(
                f"[cyan]Processing {episode_folder}...",
                total=len(png_files),
                parent=main_task,
            )

            for filename in png_files:
                input_path = os.path.join(input_episode_path, filename)
                output_path = os.path.join(output_episode_path, filename)

                rgb_to_binary_mask(input_path, output_path, target_classes)

                progress.update(
                    episode_task, advance=1, description=f"[cyan]Processed {filename}"
                )

            progress.update(
                main_task,
                advance=1,
                description=f"[bold green]Completed {episode_folder}",
            )

    console.print("[bold green]All episodes processed successfully![/bold green]")


# Example usage
input_rugd_folder = (
    "/home/moonlab/sattwik/vision-language-mapping/datasets/RUGD_annotations"
)
output_rugd_folder = "data/examples/offroad/rugd-masks"

process_rugd_dataset(
    input_rugd_folder=input_rugd_folder,
    output_rugd_folder=output_rugd_folder,
    target_classes=[
        RUGDClass.GRASS,
        RUGDClass.GRAVEL,
        RUGDClass.ASPHALT,
        RUGDClass.CONCRETE,
        RUGDClass.DIRT,
        RUGDClass.MULCH,
        RUGDClass.SAND,
        RUGDClass.PERSON
    ],
)
