import numpy as np 

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