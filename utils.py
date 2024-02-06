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


def compute_lane_width(lines):
    x1, y1, x2, y2 = lines[0]
    x3, y3, x4, y4 = lines[1]
    lane_width = np.abs((x1 + x2) / 2 - (x3 + x4) / 2)
    return lane_width

def choose_lines(lines, min_dist_x=75, return_num_lines=True):
    num_lines = len(lines)
    if lines is None or num_lines == 0:
        return None, 0
    
    if num_lines == 1:
        return lines[0], num_lines
    
    clusters = []
    num_clusters = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if not is_line_orientation_within_angle_range((x1, y1, x2, y2), 30, 150):
            continue
        # Use the midpoint or any other representative point of the line for clustering
        midpoint_x = (x1 + x2) / 2
        added_to_cluster = False
        if num_clusters == 2: # at most two clusters/lines then choose the closest one (maybe it will be better to the the longer)
            dist = []
            for cls_number in range(num_clusters):
                centroid = np.mean(clusters[cls_number], dtype=int, axis=0)
                dist.append(abs(midpoint_x - (centroid[0] + centroid[2]) / 2))
            cls = 0 if dist[0] < dist[1] else 1
            clusters[cls].append([x1, y1, x2, y2])
        else: # num_clusters < 2
            for cluster in clusters:
                if any(abs(midpoint_x - (cl[0] + cl[2]) / 2) <= min_dist_x for cl in cluster):
                    cluster.append([x1, y1, x2, y2])
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([[x1, y1, x2, y2]])
                num_clusters += 1
    
    # Generate representative lines for each cluster
    representative_lines = []
    for cluster in clusters:
        x1, y1, x2, y2 = np.mean(cluster, dtype=int, axis=0)
        representative_lines.append((x1, y1, x2, y2))

    return representative_lines, num_clusters   


def draw_lines(frame, lines, thickness=6):   
    if lines is None:
        return frame 
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
    return frame

def draw_prev_lines(frame, lines, thickness=6):   
    if lines is None:
        return frame 
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
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