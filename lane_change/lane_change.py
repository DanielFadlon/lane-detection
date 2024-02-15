import cv2
import numpy as np
from  enum import Enum 


class LaneChangeDirection(Enum):
    RIGHT = 1,
    LEFT = 2

class LaneChangeHandler:

    def __init__(self, n_first_lanes = 10):
        self.lane_width = None
        self.first_lanes_width = []
        self.required_lane_width = 0

        self.change_direction = None

        self.first_lanes_center = []
        self.current_lane_center = 0
        self.required_lane_center = 0

        self.n_first_lanes = n_first_lanes

        self.frame_counter = 0
        self.lane_changed_message_counter = 0


    def compute_lane_width(self, lines):
        x1, y1, x2, y2 = lines[0]
        x3, y3, x4, y4 = lines[1]
        lane_width = np.abs((x1 + x2) / 2 - (x3 + x4) / 2)
        return lane_width


    def compute_center(self, lines):
        x1, y1, x2, y2 = lines[0]
        x3, y3, x4, y4 = lines[1]
        center = (x1 + x2) / 2
        return center
    
    def get_current_lane_direction(self):
        return self.change_direction

    def update_lane_tracking(self, updated_lines, num_selected_lines):
        if self.change_direction is not None and self.frame_counter == 11:
            self.frame_counter = -10
            self.required_lane_width = 0
            self.first_lanes_width = []
            self.lane_width = None

        if self.frame_counter == 0 and self.change_direction is not None:
            self.change_direction = None

        if num_selected_lines == 2:
            self.lane_width = self.compute_lane_width(updated_lines)
            current_lane_center = self.compute_center(updated_lines)
            if self.frame_counter < 10:
                self.first_lanes_width.append(self.lane_width)
                self.first_lanes_center.append(self.current_lane_center)
                self.frame_counter += 1

        if self.frame_counter == 10:
            self.required_lane_width = np.mean(self.first_lanes_width)
            self.required_lane_center = np.mean(self.first_lanes_center)
            self.frame_counter += 1
    
    def detect_lane_change(self):
        if self.lane_width is not None and self.lane_width < self.required_lane_width * 0.9:
            self.lane_changed_message_counter = 400
            if self.required_lane_center < self.current_lane_center:
                self.change_direction = LaneChangeDirection.LEFT
            else:
                self.change_direction = LaneChangeDirection.RIGHT
        
        return self.change_direction is not None

    def display_lane_change_message(self, frame):
        if self.lane_changed_message_counter > 0 and self.change_direction is not None:
            if self.change_direction == self.change_direction.RIGHT:
                message = "We detected that your lane change to the right!"
            else: 
                message = "We detected that your lane change to the left!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(message, font, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, message, (text_x, text_y), font, 1, (0, 255, 0), 2)
            draw_arrow_for_lane_change(frame, self.change_direction,((text_x + 430, text_y + 50)))
            self.lane_changed_message_counter -= 1

def draw_arrow_for_lane_change(frame, lane_change_status, base_position, arrow_length=100, arrow_color=(0, 255, 0), thickness=5):
    start_point = base_position
    if lane_change_status == LaneChangeDirection.LEFT:
        end_point = (base_position[0] - arrow_length, base_position[1])
    else:
        end_point = (base_position[0] + arrow_length, base_position[1])
    
    cv2.arrowedLine(frame, start_point, end_point, arrow_color, thickness, tipLength=0.3)

