import cv2
import numpy as np
from enhancements.curve.curve import choose_lines_curve, draw_curved_lines

from lane_change.lane_direction import LaneChangeDirection
from lane_change.lane_change_utils import compute_center, compute_lane_width, display_lane_change_message
from lines.lines import choose_lines, detect_vertical_lines, draw_lines, draw_prev_lines

from enhancements.night_time.night_time import enhance_nighttime_visibility
from enhancements.vehicle_detection.vehicle_detection import detect_vehicles_in_frame

from utils import region_of_interest

# day_drive = 'data/day_drive'
day_drive = 'data/test-3-10-35'
original_night_drive = 'data/original_night_drive'
day_with_sidewalk = 'data/day_with_sidewalk'
data_type = '.mp4'



# Main function to process the video
def process_video(video_path, out_path, detect_vehicles=False, enhance_nighttime=False, detect_curve = False, 
                  width_hyper = (0.1, 0.4, 0.6, 0.9), height_hyper = (0.8, 0.6)):
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

    lane_changed_message_counter = 0

    while cap.isOpened():
      #ret: A flag that indicates whether the frame has been successfully read
        ret, frame = cap.read()
        if not ret:
            break
        frame_copy_for_car_detection = frame.copy()
        if prev_lines is not None:
            frame_with_lines = draw_prev_lines(frame, prev_lines, lane_change_status)

        if enhance_nighttime:
            gray_frame = enhance_nighttime_visibility(frame)
        else:   
        #Greyscale the frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur for smoothing out the frame
        blurred_frame = cv2.GaussianBlur(src=gray_frame, ksize=(5,5), sigmaX=0)
        # Find edges that will be point of intrest in our image
        edges = cv2.Canny(blurred_frame, 75, 220)
        vertical_lines = detect_vertical_lines(edges)
        masked_edges = region_of_interest(vertical_lines, width_hyper, height_hyper)
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=500)
        
        if lines is None:
            frame_with_lines = draw_prev_lines(frame, prev_lines, lane_change_status)
            continue
        
        updated_lines, num_selected_lines = choose_lines(lines, min_dist_x=75, return_num_lines=True)

        # Lane change detection
        if lane_change_status is not None and frame_counter == 11:
            frame_counter = -10
            required_lane_width = 0
            first_lanes_width = []
            lane_width = None

        if frame_counter == 0 and lane_change_status is not None:
            lane_change_status = None

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
        
        frame_with_lines = draw_lines(frame, updated_lines)
        prev_lines = updated_lines
 
        if lane_width is not None and lane_width < required_lane_width * 0.9:
            lane_changed_message_counter = 400
            if required_lane_center < current_lane_center:
                lane_change_status = LaneChangeDirection.LEFT
            else:
                lane_change_status = LaneChangeDirection.RIGHT

        lane_changed_message_counter = display_lane_change_message(frame, lane_changed_message_counter, lane_change_status)

        if detect_vehicles:
            detect_vehicles_in_frame(frame, frame_copy_for_car_detection)
        if detect_curve:
            lines_curve = choose_lines_curve(lines)
            frame_with_lines = draw_curved_lines(frame_with_lines, lines_curve, color=(0,0,255))

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
    # process_video(day_drive  + data_type, out_path=day_drive + '-result')
    process_video(day_drive + data_type, out_path=day_drive + '-result', detect_vehicles=True)
    #process_video(original_night_drive + data_type, out_path=original_night_drive + '-result', enhance_nighttime = True, width_hyper = (0.07, 0.4, 0.6,  0.93), height_hyper = (0.9,0.7))
    # process_video(day_with_sidewalk + data_type, out_path=day_with_sidewalk + '-result')
    print('Initiating main.py for lane detection project')