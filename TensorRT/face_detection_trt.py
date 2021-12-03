import os
import sys
import cv2
import copy
import torch
import time
from yolo_trt_model import YoloTrtModel

sys.path.insert(0, "model_face")
from utils.general import (
    check_img_size,
    non_max_suppression_face,
    scale_coords,
    xyxy2xywh,
)
from utils.datasets import letterbox


def load_model(path, face_device):
    yolo_trt_model = YoloTrtModel(face_device, path, True)
    return yolo_trt_model


def infere(input_image, yolo_trt_model, face_device):
    orgimg = input_image
    img = img_process(input_image, face_device)

    pred = yolo_trt_model(img.cpu().numpy())
    pred = yolo_trt_model.after_process(pred, face_device)
    pred = non_max_suppression_face(pred, conf_thres=0.3, iou_thres=0.5)

    # Process detections
    lt_coords = []
    left_mouth_mark = ()
    right_mouth_mark = ()
    det = pred[0]

    # normalization gain whwh
    gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(face_device)

    # normalization gain landmarks
    gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(face_device)

    # check if there is a face
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

        det[:, 5:15] = scale_coords_landmarks(
            img.shape[2:], det[:, 5:15], orgimg.shape
        ).round()

        if det.size()[0] > 1:
            # find biggest face if there are multiple faces
            biggest_box = 0
            biggest_box_arg = -1
            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                area = xywh[2] * xywh[3]
                if area > biggest_box:
                    biggest_box = area
                    biggest_box_arg = j
        else:
            biggest_box_arg = 0

        xywh = (xyxy2xywh(det[biggest_box_arg, :4].view(1, 4)) / gn).view(-1).tolist()
        landmarks = (det[biggest_box_arg, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
        lt_coords = get_left_top(input_image.shape, xywh)
        new_landmarks = []
        new_landmarks.append(
            (
                int(landmarks[6] * input_image.shape[1]),
                int(landmarks[7] * input_image.shape[0]),
            )
        )
        new_landmarks.append(
            (
                int(landmarks[8] * input_image.shape[1]),
                int(landmarks[9] * input_image.shape[0]),
            )
        )
        new_landmarks.append(
            (
                int(landmarks[4] * input_image.shape[1]),
                int(landmarks[5] * input_image.shape[0]),
            )
        )
        new_landmarks.append(
            (
                int(landmarks[2] * input_image.shape[1]),
                int(landmarks[3] * input_image.shape[0]),
            )
        )
        new_landmarks.append(
            (
                int(landmarks[0] * input_image.shape[1]),
                int(landmarks[1] * input_image.shape[0]),
            )
        )
    else:
        # no face detected
        return (0, 0)

    coords = (
        (lt_coords[0], lt_coords[1]),
        (lt_coords[0] + lt_coords[2], lt_coords[1] + lt_coords[3]),
    )
    min_y_land_arg = None
    min_y_land = 240
    for i in range(len(new_landmarks)):
        if new_landmarks[i][1] < min_y_land:
            min_y_land = new_landmarks[i][1]
            min_y_land_arg = i
    mouth_coords = []
    for i in range(len(new_landmarks)):
        if i == min_y_land_arg:
            continue
        else:
            mouth_coords.append(new_landmarks[i])

    return coords, mouth_coords


def img_process(img0, device, long_side=320, stride_max=32):
    h0, w0 = img0.shape[:2]  # orig hw
    r = long_side / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterbox(img0, new_shape=imgsz, auto=False)[0]  # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def get_left_top(im_shape, c_cord):
    x = int(im_shape[1] * (c_cord[0] - 0.5 * c_cord[2]))
    y = int(im_shape[0] * (c_cord[1] - 0.5 * c_cord[3]))
    w = int(im_shape[1] * c_cord[2])
    h = int(im_shape[0] * c_cord[3])
    return (x, y, w, h)
