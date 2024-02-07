import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import choose_lines, compute_center, compute_lane_width, draw_lines, detect_vertical_lines, region_of_interest, draw_prev_lines, LaneChanged

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

    current_lane_center = None
    prev_lines = None

    lane_width = None
    first_lanes_width = []
    required_lane_width = 0

    lane_change_status = None

    first_lanes_center = []
    required_lane_center = 0

    frame_counter = 0

    while cap.isOpened():
      #ret: A flag that indicates whether the frame has been successfully read
        ret, frame = cap.read()
        if not ret:
            break

        # if not lane_change_detected and prev_lines is not None:
        if prev_lines is not None:
            frame_with_lines = draw_prev_lines(frame, prev_lines, lane_change_status)
            
        if lane_change_status is not None and frame_counter == 11:
            frame_counter = -10
            required_lane_width = 0
            first_lanes_width = []
            lane_width = None

        if frame_counter == 0 and lane_change_status is not None:
            lane_change_status = None

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
            frame_with_lines = draw_prev_lines(frame, prev_lines, lane_change_status)
            continue
        
        updated_lines, num_selected_lines = choose_lines(lines, min_dist_x=75, return_num_lines=True)
        if num_selected_lines == 2:
            lane_width = compute_lane_width(updated_lines)
            current_lane_center = compute_center(updated_lines)
            if frame_counter < 10:
                first_lanes_width.append(lane_width)
                first_lanes_center.append(current_lane_center)
                frame_counter += 1

        if frame_counter == 10:
            required_lane_width = np.mean(first_lanes_width)
            required_lane_center = np.mean(first_lanes_center)
            frame_counter += 1
            print('Lane width:', lane_width)
        
        frame_with_lines = draw_lines(frame, updated_lines)
        prev_lines = updated_lines
 
        if lane_width is not None and lane_width < required_lane_width * 0.9:
            print('Lane width:', lane_width)
            print('Required lane width:', required_lane_width)
            print('Lane width is too small, lane change detected')
            if required_lane_center < current_lane_center:
                print('Lane change to the left')
                lane_change_status = LaneChanged.LEFT
            else:
                print('Lane change to the right')
                lane_change_status = LaneChanged.RIGHT
            # lane_change_detected = True

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