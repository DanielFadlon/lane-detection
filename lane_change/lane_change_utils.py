import cv2
import numpy as np

from lane_change.lane_direction import LaneChangeDirection 

def compute_lane_width(lines):
        x1, y1, x2, y2 = lines[0]
        x3, y3, x4, y4 = lines[1]
        lane_width = np.abs((x1 + x2) / 2 - (x3 + x4) / 2)
        return lane_width


def compute_center(lines):
    x1, y1, x2, y2 = lines[0]
    x3, y3, x4, y4 = lines[1]
    center = (x1 + x2) / 2
    return center

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
