import cv2
import numpy as np

# def mark_vehicles(frame, detections, color=(0, 0, 255), thickness=2, warning_issued = False):
#     for (x, y, w, h) in detections:
#         # Draw a rectangle around each detected vehicle
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
#         if warning_issued:
#             # Define the position for the warning text to be inside the rectangle
#             text_position = (x + 10, y + h - 10)
#             warning_message = "Too Close!"
#             cv2.putText(frame, warning_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

from collections import deque

from collections import deque
import numpy as np

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
        if i in removed_indices:
            continue

        for j, rect2 in enumerate(rectangles[i + 1:], start=i + 1):
            rect2, _ = rect2
            if j in removed_indices:
                continue

            iou = calculate_iou(rect1, rect2)
            if iou > 0.1:  # Significant overlap
                area1 = rect1[2] * rect1[3]
                area2 = rect2[2] * rect2[3]
                if area1 >= area2:
                    removed_indices.add(j)
                else:
                    removed_indices.add(i)
                    break

    for i, rect in enumerate(rectangles):
        if i not in removed_indices:
            filtered_rectangles.append(rect)

    return filtered_rectangles


MAX_BUFFER_SIZE = 200
vehicle_detection_buffer = deque(maxlen=MAX_BUFFER_SIZE)

def detect_vehicles_in_frame(frame, frame_copy_for_car_detection, min_area=2000, max_area=100000):
    global vehicle_detection_buffer
    warning_issued = False
    gray_frame = cv2.cvtColor(frame_copy_for_car_detection, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray_frame, 230, 500)
    relevant_vehicle_edges = region_of_interest_for_vehicle_detection(edges, frame)
    
    # Find contours
    contours, _ = cv2.findContours(relevant_vehicle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_frame_detections = []  # Store current frame detections
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        bottom_edge_y = y + h
        frame_height = frame.shape[0]
        ratio = w / h

        # Check if the detected contour falls within the expected area range
        if min_area < area < max_area and 0.8 < ratio < 1.5:
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


# def detect_vehicles_in_frame(frame, frame_copy_for_car_detection, min_area=4500, max_area=100000):
#     warning_issued = False
#     gray_frame = cv2.cvtColor(frame_copy_for_car_detection, cv2.COLOR_BGR2GRAY)
#     # Edge detection
#     edges = cv2.Canny(gray_frame, 175, 1000)
#     relevant_vehical_edges = region_of_interest_for_vehicle_detection(edges, frame)
#     # Morphological operations to close gaps in edges
#     # cv2.imshow('check', relevant_vehical_edges)
#     # cv2.waitKey(0)
#     # Find contours
#     contours, _ = cv2.findContours(relevant_vehical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     vehicle_detections = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         area = w * h
#         bottom_edge_y = y + h
#         frame_height = frame.shape[0]

#         # # Adjust the minimum area based on vertical position
#         # adjusted_min_area = min_area + (min_area * (bottom_edge_y / frame_height))
        
#         # Check if the detected contour falls within the expected area range
#         if min_area < area < max_area:
#             vehicle_detections.append((x, y, w, h))
            
#             # Check if the vehicle is too close based on its position or size
#             if bottom_edge_y > frame_height * 0.8 or area > (max_area * 0.5):
#                 warning_issued = True

#     mark_vehicles(frame, vehicle_detections, warning_issued)

def region_of_interest_for_vehicle_detection(edges, frame):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # trapezoid
    bottom_left = (int(width * 0.3), int(height * 0.6))
    top_left = (int(width * 0.3), int(height * 0.48))
    top_right = (int(width * 0.6), int(height * 0.48))
    bottom_right = (int(width * 0.6), int(height * 0.6))

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