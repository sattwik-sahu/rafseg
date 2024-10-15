import numpy as np
import typing as t
import cv2

def bottom_square(image: np.ndarray, threshold: float) -> t.Tuple[bool, t.Tuple[int, int, int, int]]:
    # print(f"image demensions: {image.shape}")
    #convert bgr to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is 2D
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array")

    # Get the dimensions of the image
    height, width = image.shape

    # Calculate the size of each grid cell
    cell_height = height // 4
    cell_width = width // 3

    # Calculate the coordinates for the bottom center cell
    start_row = 3 * cell_height
    end_row = height
    start_col = cell_width
    end_col = 2 * cell_width

    # Extract the bottom center cell
    bottom_center_cell = image[start_row:end_row, start_col:end_col]

    # Calculate the average value of the bottom center cell
    average_value = np.mean(bottom_center_cell)
    # print(f"average value: {average_value}")

    bottom_center_cell_coorindates = (start_row, end_row, start_col, end_col)

    # Return True if the average value is greater than the threshold, False otherwise
    return average_value > threshold, bottom_center_cell_coorindates