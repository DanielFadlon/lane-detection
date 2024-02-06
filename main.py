import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import choose_lines, compute_lane_width, draw_lines, detect_vertical_lines, region_of_interest, draw_prev_lines

# original_day_drive_with_lane_change = 'data/original_day_drive_with_lane_change'
original_day_drive_with_lane_change = 'data/lane_changed_1'
original_night_drive_with_crosswalk = 'data/original_night_drive_with_crosswalk'
data_type = '.mp4'

# Main function to process the video
def process_video(video_path, out_path, detect_sidewalk=False, detect_vehicles=False, enhance_nighttime=False):
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path + '.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    lane_change_detected = False
    current_center = None
    prev_lane_center = None
    prev_lines = None
    lane_width = None

    frame_counter = 0
    required_lane_width = 0
    first_lanes_width = []

    while cap.isOpened():
      #ret: A flag that indicates whether the frame has been successfully read
        ret, frame = cap.read()
        if not ret:
            break

        # if not lane_change_detected and prev_lines is not None:
        if not lane_change_detected and prev_lines is not None:
            frame_with_prev_lines = draw_prev_lines(frame, prev_lines)
            

        if lane_change_detected:
            frame_counter = -10
            required_lane_width = 0
            first_lanes_width = []
            lane_width = None

        if frame_counter == 0 and lane_change_detected:
            lane_change_detected = False
            continue

        # Greyscale the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply gaussian blur for smoothing out the frame
        blurred_frame = cv2.GaussianBlur(src=gray_frame, ksize=(5,5), sigmaX=0)
        # Find edges that will be point of intrest in our image
        edges = cv2.Canny(blurred_frame, 75, 220)
        vertical_lines = detect_vertical_lines(edges)
        masked_edges = region_of_interest(vertical_lines)
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=500)
        
        if lines is None:
            frame_with_lines = frame_with_prev_lines
            continue
        
        updated_lines, num_selected_lines = choose_lines(lines, min_dist_x=75, return_num_lines=True)
        if num_selected_lines == 2 and frame_counter < 10:
            lane_width = compute_lane_width(updated_lines)
            first_lanes_width.append(lane_width)
            frame_counter += 1

        if frame_counter == 10:
            required_lane_width = np.mean(first_lanes_width)
            frame_counter += 1
            print('Lane width:', lane_width)

        frame_with_lines = draw_lines(frame, updated_lines)
        prev_lines = updated_lines

        if lane_width is not None:
            print('Lane width:', lane_width)
        if lane_width is not None and lane_width < required_lane_width * 0.9:
            print('Lane width:', lane_width)
            print('Required lane width:', required_lane_width)
            print('Lane width is too small, lane change detected')
            lane_change_detected = True

        # Save the frame to a video file (this is used in order to create the submission video)
        out.write(frame_with_lines)
        # Display the image with lines
        cv2.imshow('Lane Detection', frame_with_lines)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video(original_day_drive_with_lane_change + data_type, out_path=original_day_drive_with_lane_change + '-result')
    print('Initiating main.py for lane detection project')