import cv2
from matplotlib import pyplot as plt
import numpy as np

# Helper function for selecting the region of interest
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Calculate the vertices of the trapezoid based on image size
    # These points define a trapezoid that narrows towards the top of the image, focusing on the lane area
    bottom_left = (width * 0.1, height * 0.8)  # Adjust to move the point more to the center or outward
    top_left = (width * 0.4, height * 0.6)  # Adjust to control the width of the top of the trapezoid
    top_right = (width * 0.6, height * 0.6)  # Same as above, for symmetry
    bottom_right = (width * 0.9, height * 0.8)  # Same as bottom_left, for symmetry

    # Define the polygon for the region of interest as a trapezoid
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    # Fill the specified polygon area in the mask with white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Perform a bitwise AND between the edges image and the mask to obtain the focused region of interest
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

def cluster_lines_and_draw(frame, lines, min_dist_x=25, thickness=10):
    """
    Clusters lines based on their closeness along the x-axis and draws a single representative line for each cluster.
    :param lines: Array of lines from cv2.HoughLinesP
    :param min_dist_x: Minimum distance along x-axis to consider lines as being in the same cluster.
    :return: List of representative lines for each cluster.
    """
    clusters = []
    if lines is None:
        return frame
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if not is_line_orientation_within_angle_range((x1, y1, x2, y2), 15, 165):
            continue
        # Use the midpoint or any other representative point of the line for clustering
        midpoint_x = (x1 + x2) / 2
        added_to_cluster = False
        for cluster in clusters:
            if any(abs(midpoint_x - (cl[0] + cl[2]) / 2) <= min_dist_x for cl in cluster):
                cluster.append((x1, y1, x2, y2))
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append([(x1, y1, x2, y2)])
    
    # Generate representative lines for each cluster
    for cluster in clusters:
        x1, y1, x2, y2 = np.mean(cluster, dtype=int, axis=0)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)        
    
    return frame


def draw_lines(frame, lines, segment_width=150, thickness=6):
    num_segments = frame.shape[1] // segment_width
    if lines is None:
        print('Null lines')
        plt.imshow(frame)
        return frame
    
    for segment in range(num_segments):
        # Define the x range for the current segment
        x_start = segment * segment_width
        x_end = x_start + segment_width

        max_intensity = 0
        max_line = None
        for line in lines:
            for x1, y1, x2, y2 in line:
                
                if not is_line_orientation_within_angle_range((x1, y1, x2, y2), 15, 165):
                    continue
                # Check if the line intersects with the current segment range
                if x_start <= x1 <= x_end or x_start <= x2 <= x_end:
                    # Calculate intensity or significance of the line, e.g., by its length
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                    # Update max intensity and max line if this line is more significant
                    if line_length > max_intensity:
                        max_intensity = line_length
                        max_line = (x1, y1, x2, y2)

        # If a max line was found for the segment, draw it on the image
        if max_line is not None:
            cv2.line(frame, (max_line[0], max_line[1]), (max_line[2], max_line[3]), (255, 0, 0), thickness)

    return frame


def is_line_orientation_within_angle_range(line, min_degree, max_degree):
    x1, y1, x2, y2 = line
    theta = np.arctan2(y2 - y1, x2 - x1)
    return min_degree * np.pi/180 < np.abs(theta) < max_degree * np.pi/180



def detect_vertical_lines(edges):
    # Define a vertical line convolution kernel
    kernel = np.array([[-1, 2, -1],
                    [-1, 2, -1],
                    [-1, 2, -1]])
    vertical_lines = cv2.filter2D(edges, -1, kernel)
    
    return vertical_lines