import cv2
import numpy as np

from utils import draw_lines, region_of_interest

original_day_drive_with_lane_change = 'data/original_day_drive_with_lane_change.mp4'
original_night_drive_with_crosswalk = 'data/original_night_drive_with_crosswalk.mp4'

# Main function to process the video
def process_video(video_path, detect_sidewalk=False, detect_vehicles=False, enhance_nighttime=False):
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{video_path}-result.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    global prev_lane_center
    prev_lane_center = None
    while cap.isOpened():
      #ret: A flag that indicates whether the frame has been successfully read
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally enhance visibility for nighttime
        # if enhance_nighttime:
        #     frame = enhance_nighttime_visibility(frame)

        #Greyscale the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply gaussian blur for smoothing out the frame
        blurred_frame = cv2.GaussianBlur(src=gray_frame, ksize=(5,5), sigmaX=0)
        # Find edges that will be point of intrest in our image
        edges = cv2.Canny(blurred_frame, 50, 150)
        # Use the region of intrest in order to exclude irrelevant video information
        masked_edges = region_of_interest(edges)
        # detect the lines from the masked images
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        # For each frame draw the detected lines
        frame_with_lines = draw_lines(frame, lines)

        # Optionally detect sidewalk if the flag is True
        # if detect_sidewalk:
        #     frame_with_lines = detect_crosswalk(frame_with_lines, masked_edges)

        # Optionally detect vehicles
        if detect_vehicles:
            frame = detect_vehicles(frame)
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
    process_video(original_day_drive_with_lane_change)
    print('Initiating main.py for lane detection project')