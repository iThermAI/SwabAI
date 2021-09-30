import cv2
import torch
import copy

import sys
import time

sys.path.insert(0, "model_face")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression_face,
    scale_coords,
    xyxy2xywh,
)


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


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


def infere(input_image, face_model, face_device):
    img_size = 320
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = input_image
    img0 = copy.deepcopy(orgimg)

    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=face_model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    img = torch.from_numpy(img).to(face_device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = face_model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

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

        # find biggest face if there are multiple faces
        biggest_box = 0
        biggest_box_arg = -1
        for j in range(det.size()[0]):
            xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
            area = xywh[2] * xywh[3]
            if area > biggest_box:
                biggest_box = area
                biggest_box_arg = j

        xywh = (xyxy2xywh(det[biggest_box_arg, :4].view(1, 4)) / gn).view(-1).tolist()
        conf = det[biggest_box_arg, 4].cpu().numpy()
        landmarks = (det[biggest_box_arg, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
        class_num = det[biggest_box_arg, 15].cpu().numpy()
        lt_coords = get_left_top(input_image.shape, xywh)
        left_mouth_mark = (
            int(landmarks[6] * input_image.shape[1]),
            int(landmarks[7] * input_image.shape[0]),
        )
        right_mouth_mark = (
            int(landmarks[8] * input_image.shape[1]),
            int(landmarks[9] * input_image.shape[0]),
        )
    else:
        # no face detected
        return (0, 0)

    coords = (
        (lt_coords[0], lt_coords[1]),
        (lt_coords[0] + lt_coords[2], lt_coords[1] + lt_coords[3]),
    )
    mouth_coords = (
        (left_mouth_mark[0], left_mouth_mark[1]),
        (right_mouth_mark[0], right_mouth_mark[1]),
    )
    return coords, mouth_coords


if __name__ == "__main__":
    input_image = cv2.imread("input.jpg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("face/weights/yolov5n-face.pt", device)
    infere(input_image, model, device)
