import cv2
from matplotlib import pyplot as plt
import numpy as np

# Helper function for selecting the region of interest
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Calculate the vertices of the trapezoid based on image size
    # These points define a trapezoid that narrows towards the top of the image, focusing on the lane area
    bottom_left = (width * 0.1, height * 0.8)  # Adjust to move the point more to the center or outward
    top_left = (width * 0.4, height * 0.6)  # Adjust to control the width of the top of the trapezoid
    top_right = (width * 0.6, height * 0.6)  # Same as above, for symmetry
    bottom_right = (width * 0.9, height * 0.8)  # Same as bottom_left, for symmetry

    # Define the polygon for the region of interest as a trapezoid
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    # Fill the specified polygon area in the mask with white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Perform a bitwise AND between the edges image and the mask to obtain the focused region of interest
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

def cluster_lines_and_draw(frame, lines, min_dist_x=25, thickness=10):
    """
    Clusters lines based on their closeness along the x-axis and draws a single representative line for each cluster.
    :param lines: Array of lines from cv2.HoughLinesP
    :param min_dist_x: Minimum distance along x-axis to consider lines as being in the same cluster.
    :return: List of representative lines for each cluster.
    """
    clusters = []
    if lines is None:
        return frame
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if not is_line_orientation_within_angle_range((x1, y1, x2, y2), 15, 165):
            continue
        # Use the midpoint or any other representative point of the line for clustering
        midpoint_x = (x1 + x2) / 2
        added_to_cluster = False
        for cluster in clusters:
            if any(abs(midpoint_x - (cl[0] + cl[2]) / 2) <= min_dist_x for cl in cluster):
                cluster.append((x1, y1, x2, y2))
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append([(x1, y1, x2, y2)])
    
    # Generate representative lines for each cluster
    for cluster in clusters:
        x1, y1, x2, y2 = np.mean(cluster, dtype=int, axis=0)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)        
    
    return frame


# def draw_lines(frame, lines, segment_width=150, thickness=6):
#     num_segments = frame.shape[1] // segment_width
#     if lines is None:
#         print('Null lines')
#         plt.imshow(frame)
#         return frame
    
#     for segment in range(num_segments):
#         # Define the x range for the current segment
#         x_start = segment * segment_width
#         x_end = x_start + segment_width

#         max_intensity = 0
#         max_line = None
#         for line in lines:
#             for x1, y1, x2, y2 in line:
                
#                 if not is_line_orientation_within_angle_range((x1, y1, x2, y2), 15, 165):
#                     continue
#                 # Check if the line intersects with the current segment range
#                 if x_start <= x1 <= x_end or x_start <= x2 <= x_end:
#                     # Calculate intensity or significance of the line, e.g., by its length
#                     line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#                     # Update max intensity and max line if this line is more significant
#                     if line_length > max_intensity:
#                         max_intensity = line_length
#                         max_line = (x1, y1, x2, y2)

#         # If a max line was found for the segment, draw it on the image
#         if max_line is not None:
#             cv2.line(frame, (max_line[0], max_line[1]), (max_line[2], max_line[3]), (255, 0, 0), thickness)

#     return frame


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

def detect_crosswalk(frame, edges, min_crosswalk_lines=5):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=100)
    crosswalk_line_count = 0  # Initialize count of crosswalk lines
    min_x, min_y = frame.shape[1], frame.shape[0]
    max_x, max_y = 0, 0

    if lines is not None:
        for line in lines:
            if is_crosswalk_line(line):
                crosswalk_line_count += 1
                x1, y1, x2, y2 = line[0]
                # Update bounding box coordinates
                min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
                max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)

        # If the count of lines meeting the crosswalk criteria exceeds the threshold, mark the crosswalk
        if crosswalk_line_count >= min_crosswalk_lines:
            # Draw a prominent bounding box or overlay around the crosswalk area
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 4)
            cv2.putText(frame, "Crosswalk Detected", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def is_crosswalk_line(line):
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    # Assuming crosswalk lines are mostly horizontal in the camera view
    return abs(angle) < 30 or abs(angle) > 150

def enhance_nighttime_visibility(frame):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Histogram equalization on the V channel
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    # Convert back to BGR color space
    enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_frame


def detect_lane_change_and_display_message(frame, lines, previous_lane_positions, lane_change_frames_counter):
    if lines is not None:
        current_positions = [((line[0][0] + line[0][2]) / 2, (line[0][1] + line[0][3]) / 2) for line in lines]
        current_average_position = np.mean(current_positions, axis=0) if current_positions else None

        if previous_lane_positions is not None and current_average_position is not None:
            # Calculate shift distance between current and previous positions
            shift_distance = np.linalg.norm(np.array(previous_lane_positions) - np.array(current_average_position))
            # If shift distance is above a certain threshold, consider it a lane change
            if shift_distance > 300 and lane_change_frames_counter <= 0: # Adjusted threshold and used 'and'
                lane_change_frames_counter = 30  # Display the message for the next 15 frames, adjust as needed

        # Update previous lane positions for the next frame
        previous_lane_positions = current_average_position if current_average_position is not None else previous_lane_positions

    # Display lane change message if detected or if the counter is above 0
    if lane_change_frames_counter > 0:
        cv2.putText(frame, "Lane Change Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        lane_change_frames_counter -= 1

    return previous_lane_positions, lane_change_frames_counter

def detect_significant_lane_change(frame, lines, previous_lane_positions, lane_change_frames_counter, lane_change_thresh = 550):
    if lines is None:
        lines = []  # Ensure lines is an iterable list for processing
    
    # Initialize variables to track the extreme right and left line positions
    rightmost_x = None
    leftmost_x = None

    for line in lines:
        for x1, _, x2, _ in line:
            right_x = max(x1, x2)
            left_x = min(x1, x2)
            if rightmost_x is None or right_x > rightmost_x:
                rightmost_x = right_x
            if leftmost_x is None or left_x < leftmost_x:
                leftmost_x = left_x

    # Initialize a flag for lane change detection
    lane_change_message = ""

    if 'right' in previous_lane_positions and rightmost_x is not None:
        shift_distance_right = rightmost_x - previous_lane_positions['right']
        if abs(shift_distance_right) > lane_change_thresh:
            lane_change_message = "Moved to the Left Lane" if shift_distance_right > 0 else "Moved to the Right Lane"

    if 'left' in previous_lane_positions and leftmost_x is not None:
        shift_distance_left = leftmost_x - previous_lane_positions['left']
        if abs(shift_distance_left) > lane_change_thresh:
            # Update the message only if we haven't already detected a change with the rightmost line
            # Or if the left line shift is larger, indicating a more significant maneuver.
            if not lane_change_message or abs(shift_distance_left) > abs(shift_distance_right):
                lane_change_message = "Moved to the Right Lane" if shift_distance_left < 0 else "Moved to the Left Lane"

    # Update the previous positions
    if rightmost_x is not None:
        previous_lane_positions['right'] = rightmost_x
    if leftmost_x is not None:
        previous_lane_positions['left'] = leftmost_x

    # If a lane change is detected, reset or start the lane change message counter
    if lane_change_message and lane_change_frames_counter <= 0:
        lane_change_frames_counter = 200  # Display the message for 200 frames

    # Display the lane change message if the counter is above 0
    if lane_change_frames_counter > 0:
        text_size = cv2.getTextSize(lane_change_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, lane_change_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        lane_change_frames_counter -= 1

    return previous_lane_positions, lane_change_frames_counter