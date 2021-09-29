import swab_ai
import cv2

img = cv2.imread("input.jpg")  # img is numpy array

points = swab_ai.Get_BoundingBox(img)  # returns points of bounding box

# check if points exist
if isinstance(points, int):
    if points == 0:
        # No face or mouth, No coordinations output
        print("can not detect mouth")
else:
    # save output file -> only for debugging purposes
    cv2.imwrite("output.jpg", cv2.rectangle(img, points[0], points[1], (0, 0, 255), 2))
    print("output.jpg is written.")
