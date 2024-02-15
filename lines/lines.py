import cv2
import numpy as np

from lane_change.lane_change import LaneChangeDirection
from utils import is_orientation_within_angle_range

def choose_lines(lines, min_dist_x=75):
    num_lines = len(lines)
    if lines is None or num_lines == 0:
        return None, 0
    
    if num_lines == 1:
        return lines[0], num_lines
    
    clusters = []
    num_clusters = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if not is_orientation_within_angle_range((x1, y1, x2, y2), 30, 150):
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

def draw_prev_lines(frame, lines, lane_change_status, thickness=6):   
    num_lines = len(lines)
    if lines is None or num_lines == 0:
        return frame
    
    if num_lines == 1:
        x1, y1, x2, y2 = lines[0]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
        return frame
    
    x1, y1, x2, y2 = lines[0]
    x3, y3, x4, y4 = lines[1]
    if lane_change_status == LaneChangeDirection.RIGHT:
        # draw only the previous right line
        cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), thickness)
        return frame
    if lane_change_status == LaneChangeDirection.LEFT:
        # draw only the previous left line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
        return frame
    
    
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
    cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), thickness)
    return frame

def detect_vertical_lines(edges):
    # Define a vertical line convolution kernel
    kernel = np.array([[-1, 2, -1],
                    [-1, 2, -1],
                    [-1, 2, -1]])
    vertical_lines = cv2.filter2D(edges, -1, kernel)
    
    return vertical_lines