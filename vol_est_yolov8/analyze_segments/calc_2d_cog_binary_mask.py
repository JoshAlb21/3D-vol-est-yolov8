import numpy as np
import torch

def compute_2d_cog(binary_mask: np.ndarray) -> tuple:
    """
    Compute the 2D center of gravity from a binary mask.
    
    Parameters:
    binary_mask (np.ndarray): 2D binary mask with 1s representing the object and 0s the background.
    
    Returns:
    tuple: (x_cog, y_cog) coordinates of the center of gravity.
    """
    # Ensure the input is a binary mask
    assert binary_mask.ndim == 2, "Input should be a 2D array"
    assert np.isin(binary_mask, [0, 1]).all(), "Input should be a binary mask"
    
    # Get the indices where the binary mask is 1
    y, x = np.where(binary_mask == 1)
    
    # Compute the center of gravity
    x_cog = np.mean(x)
    y_cog = np.mean(y)
    
    return (x_cog, y_cog)


def compute_cogs(masks: list, all_cls: torch.Tensor, label: dict) -> dict:
    """
    Compute the 2D center of gravity for a list of binary masks and their combinations.
    
    Parameters:
    masks (list): List of 2D binary masks with 1s representing the object and 0s the background.
    all_cls (list): List of class IDs corresponding to each mask.
    label (dict): Dictionary mapping class IDs to class labels.
    
    Returns:
    dict: Dictionary with class labels as keys and (x_cog, y_cog) as values.
    """
    cogs = {}
    combined_masks = {}

    all_cls = list(all_cls.numpy())

    # Compute CoG for each mask and save it with the class label as key
    for mask, cls in zip(masks, all_cls):
        cogs[str(int(cls))] = compute_2d_cog(mask)
    
    return cogs


if __name__ == "__main__":
    # Example binary mask: a 10x10 array with a 4x4 square of 1s at the center
    example_mask = np.zeros((10, 10), dtype=int)
    example_mask[3:7, 3:7] = 1

    # Compute the 2D CoG for the example mask
    x_cog, y_cog = compute_2d_cog(example_mask)
    print(f"2D CoG: {x_cog}, {y_cog}")
