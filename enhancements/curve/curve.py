
import cv2
import numpy as np
from utils import is_orientation_within_angle_range


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
