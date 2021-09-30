from face_detection import infere as face_infere
from face_detection import load_model as face_load_model
import dlib
from util.util import (
    resclace_coords,
    mask_bb,
    coords_or,
    draw_bbox,
    draw_point,
    show_boxes,
    check_overlap,
    landmarks2numpy,
)
from util.model_sync import sync_model
import numpy as np
import torch
import cv2
import time

# Model Weights Synchronization
sync_model()

############################## ▼ Face Inference Setup ▼ ###############################
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("No GPU is available, using cpu")
face_device = torch.device(device)
face_model = face_load_model("model_face/weights/yolov5n-face.pt", face_device)
############################## ▲ Face Inference Setup ▲ ###############################

############################## ▼ Mouth Inference Setup ▼ ##############################
Mouth_predictor = dlib.shape_predictor(
    "model_mouth/shape_predictor_68_face_landmarks.dat"
)
############################## ▲ Mouth Inference Setup ▲ ##############################
new_time = 0
prev_time = 0


def GetFace(input_image, show=False, save=False, live=False):
    """
    GetFace(input_image, show=False, save=False)

    return tuple of face image, face bounding box coordinations and mouth landmarks that detected by face detection algorthim (face_image, ((x0, y0), (x1, y1)), ((x0, y0), (x1, y1)))"""

    # inference of face detection
    coords, face_mouth_coords = face_infere(input_image, face_model, face_device)

    # check if there is a face
    if isinstance(coords, int):
        if not coords:
            # no face coordinations
            return (0, 0, 0)

    # keep face and zeropad the other pixels
    face_image = np.zeros_like(input_image)
    face_image[coords[0][1] : coords[1][1], coords[0][0] : coords[1][0]] = input_image[
        coords[0][1] : coords[1][1], coords[0][0] : coords[1][0]
    ]

    # check if save or show flags are avalable
    if save:
        cv2.imwrite("face.jpg", face_image)
    if show:
        cv2.imshow("face", face_image)
        cv2.waitKey()
    if live:
        live_face = draw_point(face_image, face_mouth_coords[0], (255, 255, 0))
        live_face = draw_point(live_face, face_mouth_coords[1], (0, 255, 255))
        cv2.imshow("face", live_face)

    return face_image, coords, face_mouth_coords


def GetMouth(face_image, face_coord, show=False, save=False, live=False):
    """
    GetMouth(face_image, show=False, save=False)

    return tuple of mouth bounding box coordinations ((x0, y0), (x1, y1))
    """

    # inference of mouth segmentation
    mask = np.zeros_like(face_image)
    face_rect = dlib.rectangle(
        face_coord[0][0], face_coord[0][1], face_coord[1][0], face_coord[1][1]
    )
    landmarks = Mouth_predictor(image=face_image, box=face_rect)
    nplands = landmarks2numpy(landmarks, 68)
    cv2.fillPoly(mask, [nplands[48:60]], color=255)

    # remove noise and find coordination of bounding box by mask
    coords = mask_bb(mask)

    # check if there is't any coordinations
    if isinstance(coords, int):
        if not coords:
            return 0

    # rescale coordinations
    coords = resclace_coords(coords, (240, 320), face_image.shape)

    # check if save or show flags are avalable
    if save:
        cv2.imwrite("mouth.jpg", mask)
    if show:
        cv2.imshow("mouth", mask)
        cv2.waitKey()
    if live:
        cv2.imshow("mouth", mask)

    return coords


def Get_BoundingBox(input_image, show=False, save=False, live=False):
    """
    Get_BoundingBox(input_image, show=False, save=False)

    returns tuple of mouth bounding box coordinations:
    ((x0, y0), (x1, y1))
    """

    # resize image
    image = cv2.resize(input_image, (320, 240))

    # prepare image for face detection model
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    face_image, face_coord, face_mouth_coords = GetFace(image, live=live)

    # check if there is't any coordinations
    if isinstance(face_coord, int):
        if not face_coord:  # no face coordinations
            print("\nFrame Dropped")
            return ()

    mouth_coords = GetMouth(face_image, face_coord, live=live)

    # check if there is't any coordinations
    if isinstance(mouth_coords, int):
        if not mouth_coords:  # no mouth coordinations
            print("\nFrame Dropped")
            return ()

    # check overlap between bbox and landmarks
    if not check_overlap(
        mouth_coords, face_mouth_coords, length_thresh=0.40, thresh_x=0.05
    ):
        print("\nFrame Dropped")
        return ()

    # rescale coordinations
    mouth_coords = resclace_coords(mouth_coords, face_image.shape, input_image.shape)

    # rescale coordinations
    face_mouth_coords = resclace_coords(
        face_mouth_coords, face_image.shape, input_image.shape
    )

    # or coordinations
    or_coords = coords_or(mouth_coords, face_mouth_coords)

    # check if save or show flags are avalable
    if show or save or live:
        # show bboxes and mouth landmarks
        input_image = show_boxes(input_image, mouth_coords, face_mouth_coords)
        if save:
            cv2.imwrite("result.jpg", input_image)
        if show:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", input_image)
            cv2.waitKey()
        if live:
            global new_time
            global prev_time
            new_time = time.time()
            fps = 1 / (new_time - prev_time)
            prev_time = new_time
            cv2.putText(
                input_image,
                str(fps),
                (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("result", input_image)

    return or_coords


############################ ▼ Load Model for first time ▼ ############################
print("Loading an empty frame for the first time -> should be dropped", end="\r")
Get_BoundingBox(np.zeros((640, 480)))  # initial loading is slower
print("All models are loaded now -> Ready ...\n")
############################ ▲ Load Model for first time ▲ ############################

if __name__ == "__main__":  # not suited for dockerized run
    image_list = []

    n = 200
    for i in range(n):
        image_list.append(
            cv2.resize(
                cv2.imread(
                    f"/mnt/DATA/Swab/Datasets/CelebAMask-HQ/CelebA-HQ-img/{i:05}.png"
                ),
                (640, 480),
            )
        )

    for i in range(n):
        Get_BoundingBox(image_list[i])
