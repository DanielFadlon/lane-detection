
import cv2
import numpy as np

def detect_and_highlight_sidewalk(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    # cv2.imshow('Sidewalk Highlighted', edges)
    # cv2.waitKey(0)
    # Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # cv2.imshow('Sidewalk Highlighted', lines)
    # cv2.waitKey(0)

    if lines is not None:
        # Initialize lists to store the coordinates of line endpoints
        x_coords = []
        y_coords = []
        
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Store the endpoints of each line
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
        
        if x_coords and y_coords:
            # Find the bounding box of the sidewalk based on the line coordinates
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            
            # Draw a rectangle around the detected sidewalk
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
            
            # Add text to label the detected sidewalk
            cv2.putText(frame, 'Sidewalk', (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)