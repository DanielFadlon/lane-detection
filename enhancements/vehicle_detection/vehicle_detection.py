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

def get_average_rect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = (w1 + w2) / 2
    h = (h1 + h2) / 2
    return (x, y, w, h)

def filter_overlapping_rectangles(rectangles):
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
            if iou > 0.3:  # Significant overlap
                  # Calculate the average dimensions
                avg_width = (rect1[2] + rect2[2]) / 2
                avg_height = (rect1[3] + rect2[3]) / 2
                
                # Determine the change for width and height
                change_width = abs(avg_width - rect2[2])
                change_height = abs(avg_height - rect2[3])
                
                # Define a threshold for considering a change as "large"
                threshold = 10  # This value is arbitrary; adjust based on your application
                
                # Check if the change is large
                if change_width > threshold or change_height > threshold:
                    # Calculate the average rectangle position
                    avg_rect_x = (rect1[0] + rect2[0]) / 2
                    avg_rect_y = (rect1[1] + rect2[1]) / 2
                    
                    # Create the average rectangle
                    avg_rect = (avg_rect_x, avg_rect_y, avg_width, avg_height)
                    
                    # Update one of the rectangles to the average rectangle
                    removed_indices.add(i)
                    removed_indices.add(j)
                    filtered_rectangles.append((0, avg_rect))
                else:
                    # Mark the other rectangle for removal
                    removed_indices.add(i)
                break
        
    for i, rect in enumerate(rectangles):
        if i not in removed_indices:
            filtered_rectangles.append(rect)

    return filtered_rectangles


MAX_BUFFER_SIZE = 100
vehicle_detection_buffer = deque(maxlen=MAX_BUFFER_SIZE)
frame_counter = 0

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
        relevant_vehicle_edges = region_of_interest_for_vehicle_detection(eroded, frame, frame_counter)
        # Find contours
        contours, _ = cv2.findContours(relevant_vehicle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_frame_detections = []  # Store current frame detections
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
        mark_vehicles(frame, detections, y_th = 0.54 * frame.shape[0])
    
    frame_counter += 1
    cv2.putText(frame, f"C: {frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def mark_vehicles(frame, vehicle_detections, y_th):
    for x, y, w, h in vehicle_detections:
        is_close_y = y + h/2 > y_th
        is_close_x = x + w/2 > frame.shape[1] * 0.38 #frame.shape[1] * 0.35 < x < frame.shape[1] * 0.7
        if is_close_y and is_close_x:
            cv2.putText(frame, "Caution!!!", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif is_close_x:
            cv2.putText(frame, "Adjacent Lane!", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)


def region_of_interest_for_vehicle_detection(edges, frame, frame_counter):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    if frame_counter < 250:
        # trapezoid
        bottom_left = (int(width * 0.3), int(height * 0.58))
        top_left = (int(width * 0.3), int(height * 0.5))
        top_right = (int(width * 0.56), int(height * 0.5))
        bottom_right = (int(width * 0.56), int(height * 0.58))
    elif frame_counter < 450:
    # trapezoid
        bottom_left = (int(width * 0.35), int(height * 0.58))
        top_left = (int(width * 0.35), int(height * 0.5))
        top_right = (int(width * 0.56), int(height * 0.5))
        bottom_right = (int(width * 0.56), int(height * 0.58))
    else:
        bottom_left = (int(width * 0.35), int(height * 0.58))
        top_left = (int(width * 0.35), int(height * 0.5))
        top_right = (int(width * 0.6), int(height * 0.5))
        bottom_right = (int(width * 0.6), int(height * 0.58))
    # Points need to be in a numpy array of shape ROWSx1x2 where ROWS is the number of vertices
    # Define the polygon for the region of interest as a trapezoid
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)
    pts = polygon.reshape((-1, 1, 2))

    # Draw the trapezoid on the image
    # cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    # Fill the specified polygon area in the mask with white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Perform a bitwise AND between the edges image and the mask to obtain the focused region of interest
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image