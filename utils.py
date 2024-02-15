import cv2
import numpy as np
from  enum import Enum

from lane_change.lane_change import LaneChangeDirection 

# Helper function for selecting the region of interest
def region_of_interest(edges, width_hyper = (0.1, 0.4, 0.6, 0.9), height_hyper = (0.8, 0.6)):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Calculate the vertices of the trapezoid based on image size
    # These points define a trapezoid that narrows towards the top of the image, focusing on the lane area
    bottom_left = (width * width_hyper[0], height * height_hyper[0])  # Adjust to move the point more to the center or outward
    top_left = (width * width_hyper[1], height * height_hyper[1])  # Adjust to control the width of the top of the trapezoid
    top_right = (width * width_hyper[2], height * height_hyper[1])  # Same as above, for symmetry
    bottom_right = (width * width_hyper[3], height * height_hyper[0])  # Same as bottom_left, for symmetry

    # Define the polygon for the region of interest as a trapezoid
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    # Fill the specified polygon area in the mask with white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Perform a bitwise AND between the edges image and the mask to obtain the focused region of interest
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

def is_orientation_within_angle_range(line, min_degree, max_degree):
    x1, y1, x2, y2 = line
    theta = np.arctan2(y2 - y1, x2 - x1)
    return min_degree * np.pi/180 < np.abs(theta) < max_degree * np.pi/180