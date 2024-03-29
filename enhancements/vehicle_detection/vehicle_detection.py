import cv2
import numpy as np
from collections import deque

from utils import region_of_interest

def calculate_iou(rect1, rect2):
    """
        Given two rectangles, this function calculates the Intersection over Union (IoU) between them.
    """
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

def get_average_rect(rect1, rect2):
    """
        Given two rectangles, this function calculates the average rectangle.
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = (w1 + w2) / 2
    h = (h1 + h2) / 2
    return (x, y, w, h)

def filter_overlapping_rectangles(rectangles):
    """
        Given a list of rectangles, this function filters out overlapping rectangles.
        Mainly, it removes rectangles that have an IoU greater than 0.3 and replaces them with an average rectangle.
    """
    filtered_rectangles = []
    removed_indices = set()
    global MAX_BUFFER_SIZE

    for i, rect1 in enumerate(rectangles):
        rect1, _ = rect1
        if i in removed_indices:
            continue
        
        for j, rect2 in enumerate(rectangles[i + 1:], start=i + 1):
            rect2, _ = rect2
            if j in removed_indices:
                continue

            iou = calculate_iou(rect1, rect2)
            if iou > 0.3:
                avg_width = (rect1[2] + rect2[2]) / 2
                avg_height = (rect1[3] + rect2[3]) / 2
                
                change_width = abs(avg_width - rect2[2])
                change_height = abs(avg_height - rect2[3])
                
                # Define a threshold for considering a change as "large"
                threshold = 10
                # Update to average rectangle if the change is large. Otherwise, remove the "old" rectangle
                if change_width > threshold or change_height > threshold:
                    avg_rect_x = (rect1[0] + rect2[0]) / 2
                    avg_rect_y = (rect1[1] + rect2[1]) / 2

                    avg_rect = (avg_rect_x, avg_rect_y, avg_width, avg_height)
                    
                    # Update one of the rectangles to the average rectangle
                    removed_indices.add(i)
                    removed_indices.add(j)
                    filtered_rectangles.append((0, avg_rect))
                else:
                    # Remove the "old" rectangle (happen when the change is not large enough)
                    removed_indices.add(i)
                break
    
    for i, rect in enumerate(rectangles):
        if i not in removed_indices:
            filtered_rectangles.append(rect)

    return filtered_rectangles


MAX_BUFFER_SIZE = 100
vehicle_detection_buffer = deque(maxlen=MAX_BUFFER_SIZE)
frame_counter = 0

def get_region(frame_counter):
    region_of_interest_hyp = {
        0: ((0.3, 0.58), (0.3, 0.5), (0.56, 0.5), (0.56, 0.58)),
        250: ((0.35, 0.58), (0.35, 0.5), (0.56, 0.5), (0.56, 0.58)),
        450: ((0.35, 0.58), (0.35, 0.5), (0.6, 0.5), (0.6, 0.58))
    }

    if frame_counter < 250:
        return region_of_interest_hyp[0]
    elif frame_counter < 450:
        return region_of_interest_hyp[250]
    else:
        return region_of_interest_hyp[450]



def detect_vehicles_in_frame(frame, frame_copy_for_car_detection):
    global vehicle_detection_buffer
    global frame_counter
    if frame_counter % 5 == 0:
        gray_frame = cv2.cvtColor(frame_copy_for_car_detection, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        # Apply gaussian blur for smoothing out the frame
        blurred_frame = cv2.GaussianBlur(src=gray_frame, ksize=(5,5), sigmaX=0)
        edges = cv2.Canny(blurred_frame, 100, 500)

        # Define kernel size
        kernel = np.ones((3,3),np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        region_hyp = get_region(frame_counter)
        relevant_vehicle_edges = region_of_interest(eroded, *region_hyp)
        # relevant_vehicle_edges = region_of_interest_for_vehicle_detection(eroded, frame, frame_counter)
        # Find contours
        contours, _ = cv2.findContours(relevant_vehicle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_frame_detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            ratio = w / h

            min_close_area = 2000
            min_far_area = 500

            is_far = y < 0.5 * frame.shape[0]
            is_close = not is_far
            is_in_required_ratio = 0.8 < ratio < 2

            if is_far and area > min_far_area and is_in_required_ratio:
                current_frame_detections.append((x, y, w, h))

            if is_close and area > min_close_area and is_in_required_ratio:
                current_frame_detections.append((x, y, w, h))
        
        # Add current frame detections to the buffer
        vehicle_detection_buffer.append(current_frame_detections)

        all_rectangles = [(rect, f_count) for f_count, frame_rects in enumerate(vehicle_detection_buffer) for rect in frame_rects]
        filtered_rectangles = filter_overlapping_rectangles(all_rectangles)
        vehicle_detection_buffer.clear()
        # reproduce the buffer with the filtered rectangles
        for i in range(MAX_BUFFER_SIZE):
            frames = [rect for rect, f_count in filtered_rectangles if f_count == i]
            vehicle_detection_buffer.append(frames)

    # Draw detections from the buffer
    for detections in vehicle_detection_buffer:
        mark_vehicles(frame, detections, y_th = 0.54, x_th = 0.38)
    
    frame_counter += 1


def mark_vehicles(frame, vehicle_detections, y_th, x_th):
    if not 0 < y_th < 1 or not 0 < x_th < 1:
        raise ValueError("y_th and x_th should be between 0 and 1")
    
    for x, y, w, h in vehicle_detections:
        is_close_y = y + h/2 > frame.shape[0] * y_th
        is_close_x = x + w/2 > frame.shape[1] * x_th
        if is_close_y and is_close_x:
            cv2.putText(frame, "Caution!!!", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)