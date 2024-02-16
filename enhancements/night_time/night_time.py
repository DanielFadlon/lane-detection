import cv2

def enhance_nighttime_visibility(frame, brightness_value = 15):
    # Convert to HSV - HSV (Hue, Saturation, Value)
    # the 'Value' channel represents brightness
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    h, s, v = cv2.split(hsv)

    # Add brightness without exceeding top limit
    lim = 255 - brightness_value
    v[v > lim] = 255
    v[v <= lim] += brightness_value

    final_hsv = cv2.merge((h, s, v))
    img_brightened = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    enhanced_frame_gray = cv2.cvtColor(img_brightened, cv2.COLOR_BGR2GRAY)
    return enhanced_frame_gray