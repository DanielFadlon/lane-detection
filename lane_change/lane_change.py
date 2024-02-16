import cv2
import numpy as np

from lane_change.lane_direction import LaneChangeDirection
from lane_change.lane_change_utils import compute_center, compute_lane_width

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
    
    def get_current_lane_direction(self):
        return self.change_direction

    def update_lane_tracking(self, updated_lines, num_selected_lines):
        if self.change_direction is not None and self.frame_counter == self.n_first_lanes + 1:
            self.frame_counter = -(self.n_first_lanes // 2)
            self.required_lane_width = 0
            self.first_lanes_width = []
            self.lane_width = None

        if self.frame_counter == 0 and self.change_direction is not None:
            self.change_direction = None

        if num_selected_lines == 2:
            self.lane_width = compute_lane_width(updated_lines)
            self.current_lane_center = compute_center(updated_lines)
            if self.frame_counter < self.n_first_lanes:
                self.first_lanes_width.append(self.lane_width)
                self.first_lanes_center.append(self.current_lane_center)
                self.frame_counter += 1

        if self.frame_counter == self.n_first_lanes:
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
        
        return self.change_direction, self.lane_changed_message_counter