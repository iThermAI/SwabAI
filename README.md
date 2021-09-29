# Swab AI
## Example Code
```python
import swab_ai
import cv2

# Reads input image using cv2
input_image = cv2.imread("./input.jpg")

# Gets points of Bounding Box by input image
points = mouth_ai.Get_BoundingBox(input_image, save=True)
```
This example code draws rectangle by using points.

output image:

![image](https://i.ibb.co/7GxNJWZ/Output.jpg)

## Python API
### `swab_ai.py`
It loads required libraries and models. This module contains three functions:
* `GetFace(input_image, show=False, save=False)`:
calls `FaceInference.py` and returns coordination of bounding box of face.
* `GetMouth(face_image, show=False, save=False)`:
calls `MouthInference.py` and returns mouth mask in (256, 256) 2D numpy array.
* `Get_BoundingBox(input_image, show=False, save=False)`:
calls `GetFace` and `GetMouth` functions and returns coordination of bounding box of mouth.

### `FaceInference.py`
Recognizes face in the images.
```python
############################## ▼ inference features ▼ ###############################
# > inference function has to return coordination of bounding box of face
# > inference function has to return coordination of mouth landmarks
# > inference function input is input image as numpy array
# > inference function has to have show_plot arg for debugging purposes
# > args: input_image, show_plot
# > outputs: ("face_image", ((x0, y0), (x1, y1)), ((xx0, yy0), (xx1, yy1)))
############################## ▲ inference features ▲ ###############################
```
### `MouthInference.py`
Recognizes mouth in the images.
```python
############################## ▼ inference features ▼ ###############################
# > inference function has to return mask in size of (256, 256)
# > inference function has to return mask in 2D numpy array
# > inference function input is input image as numpy array
# > inference function output has to be mask numpy array only
# > inference function has to have show_plot arg for debugging purposes
# > args: input_image, show_plot
# > output: mask
############################## ▲ inference features ▲ ###############################
```
## Dockerfile
Default tensorflow in this dockerfile is tensorflow:2.5.0-gpu which is the latest version now. 

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
