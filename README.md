# Swab AI
![Travis (.com)](https://img.shields.io/travis/com/ithermai/swabai)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/ithermai/swabai) 
![Lines of code](https://img.shields.io/tokei/lines/github/ithermai/swabai)
![GitHub repo size](https://img.shields.io/github/repo-size/ithermai/swabai)
![GitHub last commit](https://img.shields.io/github/last-commit/ithermai/swabai)

This code finds bounding box of a single human mouth. In comparison to other face segmentation methods, it is relatively insusceptible to open mouth conditions, e.g., yawning, surgical robots, etc. The mouth coordinates are found in a more certified way using two independent algorithms. Therefore, the algorithm can be used in more sensitive applications.

## Sample Output

![image](https://user-images.githubusercontent.com/73688480/135344354-127aacaa-676c-4262-8872-15049f23a1cb.png)
Selected images from [YawDD](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) and [CelebAMask](https://github.com/switchablenorms/CelebAMask-HQ) dataset.

## Example Code

```python
import swab_ai
import cv2

img = cv2.imread("input.jpg")  # img is numpy array

points = swab_ai.Get_BoundingBox(img)  # returns bounding box coordinates of img

if points: # bounding box is found
    # save output file -> only for debugging purposes
    cv2.imwrite("output.jpg", cv2.rectangle(img, points[0], points[1], (0, 0, 255), 2))
    print("output.jpg is written.")
```

## Python API

### `swab_ai.py`

It loads required libraries and models. This module contains three functions:

- `GetFace(input_image, show=False, save=False, live=False)`:
  calls `FaceInference.py` and returns coordinates of face bounding box and two landmarks of mouth.
- `GetMouth(face_image, show=False, save=False, live=False)`:
  return mouth mask.
- `Get_BoundingBox(input_image, show=False, save=False, live=False)`:
  calls `GetFace` and `GetMouth` functions and checks if the bounding box could be certified by checking for sufficient overlap of two independent algorithms. This function returns the mouth bounding box and also prints the FPS in the output. In case mouth bounding box could not be found `Frame Droppe` is printed in the console.

### `live_test.py`

For debugging purposes: The algorithm is run on live webcam video. The result of face detection and segmentation algorithms is visualized in real-time.

## Dockerfile

We have used the latest version of pytorch image `pytorch:1.9.0-cuda11.1-cudnn8-runtime` as of now.

To build the docker image and run the container in one step:

```shell
 docker build -t swabai .
 docker run --runtime=nvidia swabai
```

> Note that `--runtime=nvidia` is necessary to enable GPU.

If you see the message `output.jpg is written`, it means that everything is working correctly. To see the output image use this command to mount local working directory:

```shell
 docker run -v $PWD:/usr/src/app/ --runtime=nvidia swabai
```

### Model

The first time that the code is run, model is downloaded automatically. Every time that the code restarts, it checks for model updates.

## TensorRT

To use TensorRT model you have to build TensorRT engine in your device.
```bash
cd TensorRT/TRT_model
python3 onnx_to_plan.py
mv "yolov5s-exp47.plan" "../../model_face/weights/yolov5s-exp47.plan"
cd ..
mv face_detection_trt.py  swab_ai_trt.py tensorrt_com.py yolo_trt_model.py ..
```
To use TensorRT, import swab_ai_trt instead of swab_ai
