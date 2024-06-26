import numpy as np
from typing import Optional

class LengthEstimator:
    def __init__(self, points, masks, k_conv_factor: Optional[float]):
        """
        Initialize the LengthEstimator.

        Parameters:
        - points (list): List of 2D points that describe the skeleton middle line.
        - masks (dict): Dictionary with keys as object_part strings and values as binary masks.
        - k_conv_factor (float): Conversion factor to convert from pixels to mm. [mm/pixel]
        """
        self.points = np.array(points)
        self.masks = masks
        self.k_conv_factor = k_conv_factor if k_conv_factor is not None else 1

    def _calculate_length(self, mask):
        """
        Calculate the length of the line segment that falls within the given mask.

        Parameters:
        - mask (np.ndarray): Binary mask.

        Returns:
        - float: Length of the segment within the mask.
        """
        length = 0
        for i in range(1, len(self.points)):
            # Check if both the current point and the next point are within the mask
            if mask[int(self.points[i-1][1]), int(self.points[i-1][0])] == 1 and \
               mask[int(self.points[i][1]), int(self.points[i][0])] == 1:
                length += np.linalg.norm(self.points[i] - self.points[i-1])
        return length

    def calculate_lengths(self, round_to: int=1):
        """
        Calculate lengths of the skeleton line within each mask.

        Returns:
        - dict: Dictionary with keys as body parts and values as lengths.
        """
        body_part_lengths = {}
        for object_part, mask in self.masks.items():
            body_part_lengths[f'length_{object_part}'] = round(self._calculate_length(mask) * self.k_conv_factor, round_to)
        return body_part_lengths

    def calculate_total_length(self, round_to: int=1):
        """
        Calculate the total length of the skeleton line.

        Returns:
        - float: Total length of the skeleton.
        """
        total_length = np.sum([np.linalg.norm(self.points[i] - self.points[i-1]) for i in range(1, len(self.points))])
        return round(total_length * self.k_conv_factor, round_to)
