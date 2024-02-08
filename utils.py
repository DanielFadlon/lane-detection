import cv2
from matplotlib import pyplot as plt
import numpy as np
from  enum import Enum 


class LaneChanged(Enum):
    RIGHT = 1,
    LEFT = 2

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
    if lane_change_status == LaneChanged.RIGHT:
        # draw only the previous right line
        cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), thickness)
        return frame
    if lane_change_status == LaneChanged.LEFT:
        # draw only the previous left line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
        return frame
    
    
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
    cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), thickness)
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
    if lane_change_status == LaneChanged.LEFT:
        end_point = (base_position[0] - arrow_length, base_position[1])
    else:
        end_point = (base_position[0] + arrow_length, base_position[1])
    
    cv2.arrowedLine(frame, start_point, end_point, arrow_color, thickness, tipLength=0.3)


def mark_vehicles(frame, detections, color=(0, 0, 255), thickness=2, warning_issued = False):
    for (x, y, w, h) in detections:
        # Draw a rectangle around each detected vehicle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        if warning_issued:
            # Define the position for the warning text to be inside the rectangle
            text_position = (x + 5, y + h - 5)
            warning_message = "Too Close!"
            cv2.putText(frame, warning_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)


def detect_vehicles_in_frame(frame, min_area=5000, max_area=10000, aspect_ratio_range=(2.0, 3), proximity_threshold=0.75):
    warning_issued = False
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray_frame, 75, 300)
    relevant_vehical_edges = region_of_interest_for_vehicle_detection(edges)
    # Morphological operations to close gaps in edges
    
    # Find contours
    contours, _ = cv2.findContours(relevant_vehical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vehicle_detections = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Extract the coordinates of the rectangle
        x, y, w, h = cv2.boundingRect(box)
        area = w * h
        aspect_ratio = float(w) / h
        bottom_edge_y = y + h
        frame_height = frame.shape[0]
        
        # Filter based on area and aspect ratio
        if min_area < area < max_area and aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
            vehicle_detections.append((x, y, w, h))

        # Check if the vehicle is too close based on its position or size
            if bottom_edge_y > frame_height * proximity_threshold or area > (max_area * proximity_threshold):
                warning_issued = True

    mark_vehicles(frame, vehicle_detections, warning_issued)

def region_of_interest_for_vehicle_detection(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Calculate the vertices of the trapezoid based on image size
    # These points define a trapezoid that narrows towards the top of the image, focusing on the lane area
    bottom_left = (width * 0.07, height * 0.8)  # Adjust to move the point more to the center or outward
    top_left = (width * 0.4, height * 0.5)  # Adjust to control the width of the top of the trapezoid
    top_right = (width * 0.6, height * 0.5)  # Same as above, for symmetry
    bottom_right = (width * 0.85, height * 0.8)  # Same as bottom_left, for symmetry

    # Define the polygon for the region of interest as a trapezoid
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    # Fill the specified polygon area in the mask with white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Perform a bitwise AND between the edges image and the mask to obtain the focused region of interest
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

def enhance_nighttime_visibility(frame, brightness_value = 15):
    # Convert to HSV - HSV (Hue, Saturation, Value)
    # the 'Value' channel represents brightness
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    h, s, v = cv2.split(hsv)

    # Add brightness without exceeding top limit
    lim = 255 - brightness_value
    v[v > lim] = 255
    v[v <= lim] += brightness_value

    final_hsv = cv2.merge((h, s, v))
    img_brightened = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    enhanced_frame_gray = cv2.cvtColor(img_brightened, cv2.COLOR_BGR2GRAY)
    return enhanced_frame_gray
