import torch
from face_detection_trt import infere as face_infere
from face_detection_trt import load_model as face_load_model

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
import cv2
import time

# Model Weights Synchronization
# sync_model()

############################## ▼ Face Inference Setup ▼ ###############################
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("No GPU is available, using cpu")
face_device = torch.device(device)
face_model = face_load_model("model_face/weights/yolov5s-exp47.plan", face_device)
############################## ▲ Face Inference Setup ▲ ###############################

############################## ▼ Mouth Inference Setup ▼ ##############################
Mouth_predictor = dlib.shape_predictor(
    "model_mouth/shape_predictor_68_face_landmarks.dat"
)
############################## ▲ Mouth Inference Setup ▲ ##############################
new_time = 0
prev_time = 0
not_detected = 0
prev_frame = None
prev_or_coords = None
prev_face_mouth_coords = None


def GetFace(input_image, show=False, save=False, live=False):
    """
    GetFace(input_image, show=False, save=False)

    return tuple of face image, face bounding box coordinations and mouth landmarks that detected by face detection algorthim (face_image, ((x0, y0), (x1, y1)), ((x0, y0), (x1, y1)))"""

    # inference of face detection
    zeropaded_input_image = np.zeros((320, 320, 3))
    zeropaded_input_image[: input_image.shape[0], :, :] = input_image[
        : input_image.shape[0], :, :
    ]

    coords, face_mouth_coords = face_infere(
        zeropaded_input_image, face_model, face_device
    )

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
        live_face = draw_point(live_face, face_mouth_coords[2], (255, 0, 0))
        live_face = draw_point(live_face, face_mouth_coords[3], (0, 0, 255))
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


def Get_BoundingBox(
    input_image,
    show=False,
    save=False,
    live=False,
    clahe_histogram=False,
    clahe_histogram_thresh=5,
    background_subtract=True,
    background_subtract_thresh=10,
):
    """
    Get_BoundingBox(input_image, show=False, save=False)

    returns tuple of mouth bounding box coordinations:
    ((x0, y0), (x1, y1))
    """
    global prev_or_coords, prev_face_mouth_coords, prev_frame, new_time, prev_time

    # resize image
    image = cv2.resize(input_image, (320, 240))

    if clahe_histogram:
        # gray image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # clahe histogram
        clahe = cv2.createCLAHE(clipLimit=clahe_histogram_thresh)
        image = clahe.apply(image)

    if background_subtract:
        # check frames difference
        if not isinstance(prev_frame, type(None)):
            diff_frame = cv2.absdiff(
                image,
                prev_frame,
            )
            thresh_frame = cv2.threshold(
                diff_frame, background_subtract_thresh, 255, cv2.THRESH_BINARY
            )[1]
            if np.mean(thresh_frame) < background_subtract_thresh:
                if live:
                    input_image = show_boxes(
                        input_image, prev_or_coords, prev_face_mouth_coords
                    )
                    new_time = time.time()
                    fps = 1 / (new_time - prev_time)
                    prev_time = new_time
                    cv2.putText(
                        input_image,
                        str(int(fps)),
                        (7, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (100, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("result", input_image)
                    cv2.waitKey(1)
                return prev_or_coords

    # prepare image for face detection model
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    face_image, face_coord, face_mouth_coords = GetFace(image, live=live)

    # check if there is't any coordinations
    if isinstance(face_coord, int):
        if not face_coord:  # no face coordinations
            print("\nFrame Dropped")
            return ()

    x0 = min(
        face_mouth_coords[0][0],
        face_mouth_coords[1][0],
        face_mouth_coords[2][0],
        face_mouth_coords[3][0],
    )
    y0 = min(
        face_mouth_coords[0][1],
        face_mouth_coords[1][1],
        face_mouth_coords[2][1],
        face_mouth_coords[3][1],
    )
    x1 = max(
        face_mouth_coords[0][0],
        face_mouth_coords[1][0],
        face_mouth_coords[2][0],
        face_mouth_coords[3][0],
    )
    y1 = max(
        face_mouth_coords[0][1],
        face_mouth_coords[1][1],
        face_mouth_coords[2][1],
        face_mouth_coords[3][1],
    )

    face_mouth_coords = ((x0, y0), (x1, y1))

    mouth_coords = GetMouth(face_image, face_coord, live=live)

    # check if there is't any coordinations
    if isinstance(mouth_coords, int):
        if not mouth_coords:  # no mouth coordinations
            print("\nFrame Dropped")
            return ()

    # check overlap between bbox and landmarks
    if not check_overlap(
        mouth_coords,
        face_mouth_coords,
        length_thresh=0.40,
        height_thresh=0.70,
        thresh_x=0.15,
    ):
        print("\nFrame Dropped check_overlap")
        return ()

    # rescale coordinations
    mouth_coords = resclace_coords(mouth_coords, face_image.shape, input_image.shape)

    # rescale coordinations
    face_mouth_coords = resclace_coords(
        face_mouth_coords, face_image.shape, input_image.shape
    )

    # or coordinations
    or_coords = coords_or(mouth_coords, face_mouth_coords)
    prev_or_coords = or_coords
    prev_face_mouth_coords = face_mouth_coords
    if clahe_histogram:
        prev_frame = image[:, :, 0]
    else:
        prev_frame = image

    # check if save or show flags are avalable
    if show or save or live:
        # show bboxes and mouth landmarks
        input_image = show_boxes(input_image, mouth_coords, face_mouth_coords)
        if save:
            cv2.imwrite("result", input_image)
        if show:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", input_image)
            cv2.waitKey()
        if live:
            new_time = time.time()
            fps = 1 / (new_time - prev_time)
            prev_time = new_time
            cv2.putText(
                input_image,
                str(int(fps)),
                (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("result", input_image)
            cv2.waitKey(1)

    return or_coords


# ############################ ▼ Load Model for first time ▼ ############################
Get_BoundingBox(np.zeros((640, 480, 3), dtype=np.uint8))  # initial loading is slower
# ############################ ▲ Load Model for first time ▲ ############################

if __name__ == "__main__":  # not suited for dockerized run
    import os
    from tqdm import tqdm

    # image_path_list = []
    # BASE_PATH = "/mnt/DATA/@Swab/Datasets/llr_tof_images"
    # LABEL_PATH = "/home/entezari/llr_tof_images/labels/"
    # folders = [x[0] for x in os.walk(BASE_PATH)]
    # for folder in folders:
    #     files = os.listdir(folder)
    #     for file in files:
    #         if file.endswith(".png"):
    #             image_path_list.append([folder + "/" + file, LABEL_PATH + file])

    # for image_path, name in tqdm(image_path_list):
    #     if not image_path.endswith(".png"):
    #         continue
    #     try:
    #         image = cv2.imread(image_path)
    #         # print(image_path, name)
    #         image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    #         image_list.append((image, name))
    #     except Exception:
    #         continue
    #     # print(image_path, name)

    BASE_PATH = "/home/entezari/Videos/frames/"
    image_list = []
    new_time = 0
    prev_time = 0
    fps_list = []

    files = os.listdir(BASE_PATH)
    for file in files:
        if file.endswith(".jpg"):
            image = cv2.imread(BASE_PATH + file)
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
            image_list.append(image)

    for image in tqdm(image_list):
        new_time = time.time()
        fps = 1 / (new_time - prev_time)
        fps_list.append(fps)
        Get_BoundingBox(image, background_subtract=True)
        prev_time = new_time

    # print(fps_list)
    fps_list = np.array(fps_list)
    print("FPS avg:", np.average(fps_list))
    # VIDEO_PATH = "/home/entezari/Videos/frames/videoplayback.mp4"

    # new_time = 0
    # prev_time = 0
    # cap = cv2.VideoCapture(VIDEO_PATH)

    # while cap.isOpened():
    #     # Capture frame-by-frame
    #     ret, current_frame = cap.read()
    #     if type(current_frame) == type(None):
    #         print("!!! Couldn't read frame!")
    #         break

    #     # Display the resulting frame
    #     new_time = time.time()
    #     fps = 1 / (new_time - prev_time)
    #     prev_time = new_time
    #     print("HIIIIII")
    #     Get_BoundingBox(current_frame, background_subtract=True)
    #     print("FPS:", fps, "Hz    ", end="\r")

    # cap.release()

    # for i in range(n):
    #     image_list.append(
    #         cv2.resize(
    #             cv2.imread("input.jpg"),
    #             (640, 480),
    #         )
    #     )

    # image_path_list = full_image_path_list[0:5000]
    # for image_path in tqdm(image_path_list):
    #     if not image_path.endswith(".jpg"):
    #         continue
    #     image = cv2.imread(image_path)
    #     image = cv2.resize(image, (320, 240))
    #     image_list.append(image)

    # print(len(image_list))
    # for image in tqdm(image_list):
    #     Get_BoundingBox(image, live=True)

    # for i in range(n):
    #     pipeline_time0 = time.time()
    #     Get_BoundingBox(image_list[i])
    #     pipeline_time.append(time.time() - pipeline_time0)

    # face_time.pop(0)
    # pipeline_time.pop(0)
    # face_time = np.array(face_time)
    # pipeline_time = np.array(pipeline_time)
    # print("face_time avg", np.average(face_time))
    # print("pipeline_time avg", np.average(pipeline_time))
    # print("face_time min", np.min(face_time))
    # print("pipeline_time min", np.min(pipeline_time))
    # import os
    # from tqdm import tqdm

    # BASE_PATH = "/mnt/DATA/@Swab/Datasets/CelebAMask-HQ/CelebA-HQ-img/"
    # # BASE_PATH = "/mnt/DATA/@Swab/Datasets/OpenMouth-320x240/front/images/"
    # # OUTPUT_PATH = "/home/entezari/swab/openmouth320/front/"
    # files = os.listdir(BASE_PATH)
    # image_path_list = [BASE_PATH + file for file in files]
    # image_path_list = image_path_list[0:5000]
    # image_list = []
    # no_detected = 0

    # for image_path in tqdm(image_path_list):
    #     image_list.append(cv2.imread(image_path))

    # time0 = time.time()
    # for i, image in enumerate(image_list):
    #     points = Get_BoundingBox(
    #         image, background_subtract=False, clahe_histogram=False
    #     )
    #     if points:  # bounding box is found
    #         # cv2.imwrite(
    #         #     OUTPUT_PATH + str(i) + ".png",
    #         #     cv2.rectangle(image, points[0], points[1], (0, 0, 255), 2),
    #         # )
    #         pass
    #     else:
    #         no_detected += 1
    # time1 = time.time()
    # print("time", (time1 - time0) / 5000)
    # print("not detected:", no_detected)
