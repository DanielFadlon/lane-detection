import cv2
import numpy as np
from collections import deque

def calculate_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area
    return intersection_area / union_area

def filter_overlapping_rectangles(rectangles):
    filtered_rectangles = []
    removed_indices = set()

    for i, rect1 in enumerate(rectangles):
        rect1, _ = rect1
        aspect_ratio_1 = (rect1[2] - rect1[0]) / (rect1[3] - rect1[1])
        if i in removed_indices:
            continue

        for j, rect2 in enumerate(rectangles[i + 1:], start=i + 1):
            rect2, _ = rect2
            aspect_ratio_2 = (rect2[2] - rect2[0]) / (rect2[3] - rect2[1])
            if j in removed_indices:
                continue

            iou = calculate_iou(rect1, rect2)
            if iou > 0.15:  # Significant overlap
                area1 = rect1[2] * rect1[3]
                area2 = rect2[2] * rect2[3]
                offset = 0.2
                if aspect_ratio_1 + offset < aspect_ratio_2:
                    removed_indices.add(j)
                elif aspect_ratio_2 + offset < aspect_ratio_1:
                    removed_indices.add(i)
                elif area1 >= area2:
                    removed_indices.add(j)
                elif area2 >= area1:
                    removed_indices.add(i)
                else:
                    removed_indices.add(i)
                    break

    for i, rect in enumerate(rectangles):
        if i not in removed_indices:
            filtered_rectangles.append(rect)

    return filtered_rectangles


MAX_BUFFER_SIZE = 100
vehicle_detection_buffer = deque(maxlen=MAX_BUFFER_SIZE)

def detect_vehicles_in_frame(frame, frame_copy_for_car_detection, min_area=1500, max_area=6000):
    global vehicle_detection_buffer
    warning_issued = False
    gray_frame = cv2.cvtColor(frame_copy_for_car_detection, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    # Apply gaussian blur for smoothing out the frame
    blurred_frame = cv2.GaussianBlur(src=gray_frame, ksize=(5,5), sigmaX=0)
    edges = cv2.Canny(blurred_frame, 100, 500)

    # Define kernel size
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    relevant_vehicle_edges = region_of_interest_for_vehicle_detection(eroded, frame)

    
    # Find contours
    # contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(relevant_vehicle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_frame_detections = []  # Store current frame detections
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        bottom_edge_y = y + h
        frame_height = frame.shape[0]
        ratio = w / h

        # Check if the detected contour falls within the expected area range
        if min_area < area < max_area and ratio < 2:
            current_frame_detections.append((x, y, w, h))
            
            # Check if the vehicle is too close based on its position or size
            if bottom_edge_y > frame_height * 0.8 or area > (max_area * 0.5):
                warning_issued = True
    
    # Add current frame detections to the buffer
    vehicle_detection_buffer.append(current_frame_detections)

    # Consolidate all rectangles into a single list
    all_rectangles = [(rect, f_count) for f_count, frame_rects in enumerate(vehicle_detection_buffer) for rect in frame_rects]

    # Filter the consolidated list of rectangles
    filtered_rectangles = filter_overlapping_rectangles(all_rectangles)

    # If you need to clear the deque and replace it with filtered rectangles
    # Note: This step will depend on how you wish to use the filtered results
    vehicle_detection_buffer.clear()
    # reproduce the buffer with the filtered rectangles
    for i in range(MAX_BUFFER_SIZE):
        frames = [rect for rect, f_count in filtered_rectangles if f_count == i]
        vehicle_detection_buffer.append(frames)
    
    # Draw detections from the buffer
    for detections in vehicle_detection_buffer:
        mark_vehicles(frame, detections)

def mark_vehicles(frame, vehicle_detections):
    # Your existing logic to mark vehicles on the frame
    # Add any additional logic here if needed to handle the warning_issued flag
    for x, y, w, h in vehicle_detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)


def region_of_interest_for_vehicle_detection(edges, frame):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # trapezoid
    bottom_left = (int(width * 0.3), int(height * 0.58))
    top_left = (int(width * 0.3), int(height * 0.5))
    top_right = (int(width * 0.56), int(height * 0.5))
    bottom_right = (int(width * 0.56), int(height * 0.58))

    # Points need to be in a numpy array of shape ROWSx1x2 where ROWS is the number of vertices
    # Define the polygon for the region of interest as a trapezoid
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)
    pts = polygon.reshape((-1, 1, 2))

    # Draw the trapezoid on the image
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    # Fill the specified polygon area in the mask with white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Perform a bitwise AND between the edges image and the mask to obtain the focused region of interest
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image