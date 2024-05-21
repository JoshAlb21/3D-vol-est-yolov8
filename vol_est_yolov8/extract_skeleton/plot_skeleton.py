import cv2
import numpy as np
import os

class LineDrawer:
    def __init__(self, points: np.ndarray, image: np.ndarray, bin_mask: bool = False, scatter_points: np.ndarray = None, orthogonal_lines: (list, dict) = None, conv_2_rgb: bool = True):
        self.points = points.astype(int)  # Convert points to integer
        self.scatter_points = scatter_points.astype(int) if scatter_points is not None else None
        self.orthogonal_lines = orthogonal_lines
        self.bin_mask = bin_mask
        if bin_mask:
            self.image = self._prepare_binary_mask(image)
            self.line_color = (255, 0, 0)  # Blue line for BGR when bin_mask is True
        else:
            if conv_2_rgb:
                image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
            self.image = image
            self.line_color = (0, 0, 255)  # Red line for BGR when bin_mask is False
        self.marker_color = (0, 255, 0)  # Green marker color
        self.scatter_color = (255, 0, 0)  # Blue color for scatter points
        if isinstance(self.orthogonal_lines, dict):
            predefined_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Blue, Red in BGR format 
            self.segment_colors = {seg: predefined_colors[i % len(predefined_colors)] for i, seg in enumerate(orthogonal_lines)}
        else:
            self.ortho_line_color = (255, 255, 0)  # Default yellow color for orthogonal lines if it's a list

    def _prepare_binary_mask(self, mask: np.ndarray):
        """Prepare binary mask for visualization."""
        if not isinstance(mask, np.ndarray):
            raise TypeError("Input mask should be a NumPy array")
        if mask.ndim != 2:
            raise ValueError("Input mask should be a 2D array")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Input mask should be binary (containing only 0s and 1s)")
        
        mask = mask.astype(np.uint8)  # Ensure it's uint8
        mask *= 255  # Scale from [0, 1] to [0, 255] if necessary
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
        return mask

    def draw_line(self):
        for i in range(1, len(self.points)):
            pt1 = tuple(self.points[i - 1])
            pt2 = tuple(self.points[i])
            cv2.line(self.image, pt1, pt2, self.line_color, 10)
        self._draw_start_end_markers()

    def draw_scatter(self):
        if self.scatter_points is not None:
            for pt in self.scatter_points:
                cv2.circle(self.image, tuple(pt), 5, self.scatter_color, -1)
    
    def draw_orthogonal_lines(self):
        if isinstance(self.orthogonal_lines, list):
            for line in self.orthogonal_lines:
                for i in range(1, len(line)):
                    pt1 = (int(line[i - 1][0]), int(line[i - 1][1]))
                    pt2 = (int(line[i][0]), int(line[i][1]))
                    cv2.line(self.image, pt1, pt2, self.ortho_line_color, 2)
        elif isinstance(self.orthogonal_lines, dict):
            for seg, lines in self.orthogonal_lines.items():
                # Convert numpy.int64 values to standard Python int
                color = tuple(map(int, self.segment_colors[seg]))
                for line in lines:
                    for i in range(1, len(line)):
                        pt1 = (int(line[i - 1][0]), int(line[i - 1][1]))
                        pt2 = (int(line[i][0]), int(line[i][1]))
                        cv2.line(self.image, pt1, pt2, color, 2)

    def _draw_start_end_markers(self):
        start_point = self.points[0]
        end_point = self.points[-1]
        self._draw_perpendicular_line(start_point, self.points[1])
        self._draw_perpendicular_line(end_point, self.points[-2])

    def _draw_perpendicular_line(self, point, adjacent_point):
        dx = adjacent_point[0] - point[0]
        dy = adjacent_point[1] - point[1]
        length = 50  # Length of the perpendicular marker lines

        # Calculate coordinates for the marker lines
        if dx != 0 and dy != 0:  # Avoid division by zero
            slope = -dx / dy
            # Compute the angle of the perpendicular line to the x-axis
            angle = np.arctan(slope)
            x_offset = length * np.cos(angle)
            y_offset = length * np.sin(angle)
        elif dx == 0:  # Original line is vertical
            x_offset = length
            y_offset = 0
        else:  # Original line is horizontal
            x_offset = 0
            y_offset = length
            
        x1 = int(point[0] - x_offset)
        y1 = int(point[1] - y_offset)
        x2 = int(point[0] + x_offset)
        y2 = int(point[1] + y_offset)
        
        cv2.line(self.image, (x1, y1), (x2, y2), self.marker_color, 20)

    def show_image(self):
        self.draw_orthogonal_lines()  # Call this method before the other drawings
        self.draw_scatter()  # Call this method before showing the image
        cv2.imshow('Image with Line', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_img(self):
        self.draw_line()
        self.draw_orthogonal_lines()  # Call this method before the other drawings
        self.draw_scatter()  # Call this method before showing the image
        return self.image

    def save_image(self, path: str, name:str):
        if not name.endswith(".png") and not name.endswith(".jpg"):
            raise NotImplementedError(f"Name should end with .png or .jpg, File extension not supported: {name.split('.')[-1]}")
        self.draw_scatter()
        path_to_save = os.path.join(path, name)
        cv2.imwrite(path_to_save, self.image)


if __name__ == '__main__':
    # Usage example:
    points = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a black image
    drawer = LineDrawer(points, image)
    drawer.draw_line()
    drawer.show_image()