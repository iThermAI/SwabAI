import swab_ai
import cv2

img = cv2.imread("input.jpg")  # img is numpy array

points = swab_ai.Get_BoundingBox(img)  # returns bounding box coordinates of img

if points:  # bounding box is found
    # save output file -> only for debugging purposes
    cv2.imwrite("output.jpg", cv2.rectangle(img, points[0], points[1], (0, 0, 255), 2))
    print("output.jpg is written.")
