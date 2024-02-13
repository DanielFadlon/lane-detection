import cv2
from matplotlib import pyplot as plt
import numpy as np
from  enum import Enum 
i=0

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


def draw_lines(frame, lines, thickness=6, color = (255, 0, 0)):   
    if lines is None:
        return frame 
    for line in lines:
        if isinstance(line, int):
            continue
        if  len(line) != 4:
            continue
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def draw_prev_lines(frame, lines, lane_change_status, thickness=6, color = (255, 0, 0)):   
    num_lines = len(lines)
    if lines is None or num_lines == 0:
        return frame
    
    if num_lines == 1:
        x1, y1, x2, y2 = lines[0]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
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
            text_position = (x + 10, y + h - 10)
            warning_message = "Too Close!"
            cv2.putText(frame, warning_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)


def detect_vehicles_in_frame(frame, frame_copy_for_car_detection, min_area=7500, max_area=100000):
    global i
    i+=1
    warning_issued = False
    gray_frame = cv2.cvtColor(frame_copy_for_car_detection, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray_frame, 175, 1000)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        bottom_edge_y = y + h
        frame_height = frame.shape[0]

        # # Adjust the minimum area based on vertical position
        # adjusted_min_area = min_area + (min_area * (bottom_edge_y / frame_height))
        
        # Check if the detected contour falls within the expected area range
        if min_area < area < max_area:
            vehicle_detections.append((x, y, w, h))
            
            # Check if the vehicle is too close based on its position or size
            if bottom_edge_y > frame_height * 0.8 or area > (max_area * 0.5):
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

def choose_lines_curve(lines, min_dist_x=75, extension_factor=200):
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
        x1_new, y1_new, x2_new, y2_new = extend_line(x1, y1, x2, y2, extension_factor)
        representative_lines.append((int(x1_new), int(y1_new), int(x2_new), int(y2_new)))

    return representative_lines   

def extend_line(x1, y1, x2, y2, extension_factor):
    # Calculate the midpoint of the original line
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Calculate the direction vector of the line
    direction = np.array([x2 - x1, y2 - y1])
    norm_direction = direction / np.linalg.norm(direction)
    
    # Extend the start and end points
    x1_new, y1_new = mid_x - norm_direction[0] * extension_factor , mid_y - norm_direction[1] * extension_factor
    x2_new, y2_new = mid_x + norm_direction[0] * extension_factor , mid_y + norm_direction[1] * extension_factor
    
    return x1_new, y1_new, x2_new, y2_new

# def draw_curved_lines(frame, lines, thickness=6, color = (255, 0, 0)):   
#     if lines is None:
#         return frame 
#     for i in range(len(lines) - 1):
#         if isinstance(lines[i], int) or isinstance(lines[i+1], int):
#             continue
#         if  len(lines[i]) != 4 and len(lines[i+1]) != 4:
#             continue
#         x1, y1, x2, y2 = lines[i]
#         x3, y3, x4, y4 = lines[i+1]
#         x5 = int((x2+x3)/2)
#         x6 = int((x1+x4)/2)
#         y5 = int((y2+y3)/2) 
#         y6 = int((y1+y4)/2)
#         cv2.line(frame, (x5, y5), (x6, y6), color, thickness)
#     return frame

def draw_curved_lines(frame, lines, color=(255, 0, 0)):
    if lines is None:
        return frame
    for i in range(len(lines) - 1):
        if isinstance(lines[i], int) or isinstance(lines[i+1], int):
            continue
        if len(lines[i]) != 4 or len(lines[i+1]) != 4:
            continue
        x1, y1, x2, y2 = lines[i]
        x3, y3, x4, y4 = lines[i+1]

        # Calculating midpoints of the ends of the two line segments
        midpoint1 = ((x2 + x3) / 2, (y2 + y3) / 2)
        midpoint2 = ((x1 + x4) / 2, (y1 + y4) / 2)

        # Control point as the average of all four points
        if x2 - x1 - 337 > 0:
            angle_adj = 50
        elif x2 - x1 + 300 < 0:
            angle_adj = -50
        else:
           angle_adj = 0 
        control_point = ((x1 + x2 + x3 + x4) / 4 + angle_adj, (y1 + y2 + y3 + y4) / 4 - 10)
        # Drawing the Bezier curve
        draw_bezier_curve(frame, midpoint1, control_point, midpoint2, color)

    return frame

# def draw_bezier_curve(img, p0, p1, p2, color=(255, 0, 0), thickness=20):
#     # Generate points for the Bezier curve
#     # A quadratic Bezier curve is defined by three points: a start point (P0), 
#     # a control point (P1), and an end point (P2).
#     # For a given t in [0, 1], the curve point is calculated as:
#     # (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
#     # Where P0 is the start point, P1 is the control point, and P2 is the end point.
#     points = []
#     for t in np.linspace(0, 1, num=100):
#         point = (1-t)**2*np.array(p0) + 2*(1-t)*t*np.array(p1) + t**2*np.array(p2)
#         points.append(point.astype(int))
#     points = np.array(points, np.int32)
#     cv2.polylines(img, [points], False, color, thickness)

def draw_bezier_curve(img, p0, p1, p2, color=(255, 0, 0), thickness=2, dot_size=10, space_size=10):
    # Generate points for the Bezier curve
    points = []
    for t in np.linspace(0, 1, num=50):
        point = (1-t)**2*np.array(p0) + 2*(1-t)*t*np.array(p1) + t**2*np.array(p2)
        points.append(point.astype(int))
    points = np.array(points, np.int32)
    
    # Draw dots along the Bezier curve
    for i, point in enumerate(points):
        if i % (dot_size + space_size) < dot_size:
            cv2.circle(img, tuple(point), thickness, color, -1)

    text_position = (int(p1[0])-100, int(p1[1] - 20))
    
    # Draw the text on the image
    cv2.putText(img, "Expected Curvature", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)