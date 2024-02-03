import cv2

original_day_drive_with_lane_change = 'data/original_day_drive_with_lane_change.mp4'
original_night_drive_with_crosswalk = 'data/original_night_drive_with_crosswalk.mp4'

if __name__ == '__main__':
    cap = cv2.VideoCapture(original_day_drive_with_lane_change)
    print('Initiating main.py for lane detection project')