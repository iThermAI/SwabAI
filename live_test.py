import cv2
import numpy as np
import swab_ai
import time

new_time = 0
prev_time = 0
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, current_frame = cap.read()
    if type(current_frame) == type(None):
        print("!!! Couldn't read frame!")
        break

    # Display the resulting frame
    new_time = time.time()
    fps = 1 / (new_time - prev_time)
    prev_time = new_time
    swab_ai.Get_BoundingBox(current_frame, live=True)
    print("FPS:", fps, "Hz    ", end="\r")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
