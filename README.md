# Swab AI
## Example Code
```python
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
```

Output image examples with enabled save flag:

![image](https://user-images.githubusercontent.com/73688480/135344354-127aacaa-676c-4262-8872-15049f23a1cb.png)


## Python API
### `swab_ai.py`
It loads required libraries and models. This module contains three functions:
* `GetFace(input_image, show=False, save=False, live=False)`:
calls `FaceInference.py` and returns coordination of bounding box of face and two landmarks of mouth.
* `GetMouth(face_image, show=False, save=False, live=False)`:
return mask of mouth by using dlib library.
* `Get_BoundingBox(input_image, show=False, save=False, live=False)`:
calls `GetFace` and `GetMouth` functions and check if overlap area of two methods is accepted by conditions or not,then by the result returns coordination of bounding box of mouth or print Frame Dropped.

### `live_test.py`
Uses webcam frames as the input image and shows cropped face image and mouth mask and output image with bounding boxes. The yellow and light blue points are face algorithm landmarks, the red rectangle is mouth algorithm bounding box and the green rectangle is final coordinations output.

## Dockerfile
Default pytorch in this dockerfile is pytorch:1.9.0-cuda11.1-cudnn8-runtime which is the latest version now. 

To make and run the docker image in one step: 
```bash
 docker build -t swabai . 
 docker run --runtime=nvidia swabai
```
If you see the message `output.jpg is written`, it means that everything is working correctly. To see the output image use this command to mount it:
```bash
 docker run -v $PWD:/usr/src/app/ --runtime=nvidia swabai
```

### Model
The first time that the code is run, model is downloaded automatically. Every time that the code restarts, it checks for model updates.

