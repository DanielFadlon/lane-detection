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



def display_lane_change_message(frame, message_counter, lane_change_status):
    if message_counter > 0 and lane_change_status is not None:
        if lane_change_status == lane_change_status.RIGHT:
            message = "We detected that your lane change to the right!"
        else: 
            message = "We detected that your lane change to the left!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(message, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, message, (text_x, text_y), font, 1, (0, 255, 0), 2)
        draw_arrow_for_lane_change(frame, lane_change_status,((text_x + 430, text_y + 50)))
        message_counter -= 1
    return message_counter

def draw_arrow_for_lane_change(frame, lane_change_status, base_position, arrow_length=100, arrow_color=(0, 255, 0), thickness=5):
    start_point = base_position
    if lane_change_status == LaneChangeDirection.LEFT:
        end_point = (base_position[0] - arrow_length, base_position[1])
    else:
        end_point = (base_position[0] + arrow_length, base_position[1])
    
    cv2.arrowedLine(frame, start_point, end_point, arrow_color, thickness, tipLength=0.3)



