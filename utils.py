import cv2
import numpy as np
from  enum import Enum

def region_of_interest(edges, bottom_left_hyp, top_left_hyp, top_right_hyp, bottom_right_hyp):
    """
    Applies an image mask for selecting a region of interest.
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Calculate the vertices of the trapezoid based on image size
    bottom_left = (width * bottom_left_hyp[0], height * bottom_left_hyp[1])
    top_left = (width * top_left_hyp[0], height * top_left_hyp[1])
    top_right = (width * top_right_hyp[0], height * top_right_hyp[1])
    bottom_right = (width * bottom_right_hyp[0], height * bottom_right_hyp[1]) 

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