import torch


def calculate_binary_iou(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
):
    """
    Calculates IoU for binary segmentation (0 and 1 values).

    Args:
        pred (torch.Tensor): Predicted segmentation map (HxW).
        target (torch.Tensor): Ground truth segmentation map (HxW), with values 0 and 255.
        threshold (float): Threshold to binarize the predicted output, default is 0.5.

    Returns:
        float: IoU score.
    """

    # Convert ground truth from {0, 255} to {0, 1}
    target = (target == 255).int()

    # Apply threshold to predictions if needed (assuming pred is not yet binary)
    pred = (pred > threshold).int()

    # Calculate intersection and union
    intersection = (pred & target).sum().float().item()  # True positives
    union = (
        (pred | target).sum().float().item()
    )  # True positives + False positives + False negatives

    # Avoid division by zero
    if union == 0:
        return float("nan")  # Return NaN if both pred and target are empty
    else:
        iou = intersection / union

    return iou
