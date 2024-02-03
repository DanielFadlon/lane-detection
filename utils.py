import cv2
import numpy as np

# Helper function for selecting the region of interest
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define the polygon for the region of interest
    # The polygon is defined to cover the bottom part of the image, starting from 80% of the height
    # This effectively focuses the region of interest on the lower part of the image where the lanes on the road are more likely to be found,
    # excluding the upper part of the image which might contain irrelevant details.
    polygon = np.array([[(0, height * 0.8), (width, height * 0.8), (width, height), (0, height)],], np.int32)

    #This function fills the specified polygon area in the mask with the color white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Performs a bitwise AND operation between the edges image and the mask
    # only the edges that fall within the region of interest are visible in the masked_image, and everything else is blacked out
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

# Helper function for drawing Hough lines
# def draw_lines(frame, lines, thickness=3):
#     global prev_lane_center
#     current_lane_center = 0
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
#                 current_lane_center += (x1 + x2) / 2
#         if len(lines) > 0:
#             current_lane_center /= len(lines)  # Average center for current frame

#     # Detect lane change based on the shift of the lane center
#     # if prev_lane_center is not None:
#     #     lane_change_detected(current_lane_center, frame)

#     # Update for next frame
#     prev_lane_center = current_lane_center
#     return frame


import cv2
import numpy as np

def draw_lines(frame, lines, thickness=10):
    global prev_lane_center
    current_lane_center = 0
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate slope and angle
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                angle = np.arctan(slope) * 180 / np.pi
                
                # Filter lines based on angle threshold, e.g., within +/- 30 degrees of vertical
                angle_threshold = 70  # adjust as necessary
                if abs(angle) > (90 - angle_threshold) and abs(angle) < (90 + angle_threshold):
                    # Make lines longer
                    if slope != 0:  # Avoid division by zero
                        # Extend to the bottom of the region of interest
                        y1_new = int(frame.shape[0])
                        x1_new = int(x1 + (y1_new - y1) / slope)
                        # Extend to the top of the region of interest (or to a fixed point)
                        y2_new = int(frame.shape[0] * 0.8)
                        x2_new = int(x1 + (y2_new - y1) / slope)
                        
                        # Draw the extended line
                        cv2.line(frame, (x1_new, y1_new), (x2_new, y2_new), (255, 0, 0), thickness)
                        current_lane_center += (x1_new + x2_new) / 2
        if len(lines) > 0:
            current_lane_center /= len(lines)  # Average center for current frame

    # Update for next frame
    prev_lane_center = current_lane_center
    return frame




def detect_vertical_lines(edges):
    # Define a vertical line convolution kernel
    # This kernel is designed to highlight vertical lines and suppress horizontal ones.
    kernel = np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]])
    
    # Apply convolution using the defined kernel
    # cv2.filter2D requires a source image, desired depth (-1 to use the same as source), and the kernel
    convolved_edges = cv2.filter2D(edges, -1, kernel)
    
    return convolved_edges