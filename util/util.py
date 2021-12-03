import cv2
import numpy as np


def resclace_coords(input_coords, input_shape, output_shape):
    x0 = int(input_coords[0][0] * output_shape[1] / input_shape[1])
    y0 = int(input_coords[0][1] * output_shape[0] / input_shape[0])
    x1 = int(input_coords[1][0] * output_shape[1] / input_shape[1])
    y1 = int(input_coords[1][1] * output_shape[0] / input_shape[0])
    coords = ((x0, y0), (x1, y1))
    return coords


def mask_bb_np(mask):
    mask = mask[:, :, 0]
    maskx = np.any(mask, axis=0)
    masky = np.any(mask, axis=1)
    x0 = np.argmax(maskx)
    y0 = np.argmax(masky)
    x1 = len(maskx) - np.argmax(maskx[::-1])
    y1 = len(masky) - np.argmax(masky[::-1])
    coords = ((x0, y0), (x1, y1))
    return coords


def coords_or2(mouth_coords, face_mouth_coords):
    return (
        (
            min(face_mouth_coords[0][0], mouth_coords[0][0]),
            min(face_mouth_coords[0][1], face_mouth_coords[1][1], mouth_coords[0][1]),
        ),
        (
            max(face_mouth_coords[1][0], mouth_coords[1][0]),
            max(face_mouth_coords[0][1], face_mouth_coords[1][1], mouth_coords[1][1]),
        ),
    )


def coords_or(mouth_coords, face_mouth_coords):
    return (
        (
            min(face_mouth_coords[0][0], mouth_coords[0][0]),
            min(face_mouth_coords[0][1], mouth_coords[0][1]),
        ),
        (
            max(face_mouth_coords[1][0], mouth_coords[1][0]),
            max(face_mouth_coords[1][1], mouth_coords[1][1]),
        ),
    )


def draw_bbox(image, coords, color=(0, 0, 255), thick=2):
    return cv2.rectangle(image, coords[0], coords[1], color, thick)


def draw_point(image, coord, color=(0, 0, 255)):
    return cv2.circle(image, coord, 4, color, -1)


def show_boxes(image, mouth_coords, face_mouth_coords):
    or_coords = coords_or(mouth_coords, face_mouth_coords)
    image = draw_bbox(image, mouth_coords)
    image = draw_bbox(image, or_coords, (0, 255, 0), 1)
    return image


def remove_noise_return_contour(mask, kernel_size=(5, 5), area_thresh=0.5):

    # opening filter
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # get contours
    thresh = cv2.Canny(opening, 100, 200)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 1:
        # multiple contours
        area_list = [cv2.contourArea(cnt) for cnt in contours]
        area_arg_list = np.argsort(area_list)[::-1]

        if not area_list[area_arg_list[0]] * area_thresh <= area_list[area_arg_list[1]]:
            # second biggest contour is bigger than threshold, can't find proper contour
            return 0

        # found proper contour and return it
        return contours[area_arg_list[0]]
    elif len(contours) == 0:
        # no contour
        return 0
    else:
        # one contour
        return contours[0]


def mask_bb(mask):
    mask = mask[:, :, 0]
    maskx = np.any(mask, axis=0)
    masky = np.any(mask, axis=1)
    x0 = np.argmax(maskx)
    y0 = np.argmax(masky)
    x1 = len(maskx) - np.argmax(maskx[::-1])
    y1 = len(masky) - np.argmax(masky[::-1])
    coords = ((x0, y0), (x1, y1))
    return coords


def center_mass(coords):
    cmx = abs(coords[1][0] - coords[0][0]) / 2
    cmy = abs(coords[1][1] - coords[0][1]) / 2
    return (cmx, cmy)


def coords_avg2(box_coords, line_coords):
    box_x = abs(box_coords[1][0] - box_coords[0][0])
    box_y = abs(box_coords[1][1] - box_coords[0][1])
    line_x = abs(line_coords[1][0] - line_coords[0][0])
    avg_x = (box_x + line_x) / 2
    avg_y = box_y
    return (avg_x, avg_y)


def coords_avg(dlib_coords, yolo_coords):
    dlib_x = abs(dlib_coords[1][0] - dlib_coords[0][0])
    dlib_y = abs(dlib_coords[1][1] - dlib_coords[0][1])
    yolo_x = abs(yolo_coords[1][0] - yolo_coords[0][0])
    yolo_y = abs(yolo_coords[1][1] - yolo_coords[0][1])
    avg_x = (dlib_x + yolo_x) / 2
    avg_y = (dlib_y + yolo_y) / 2
    return (avg_x, avg_y)


def check_overlap2(
    box_coords, line_coords, thresh_x=0.05, thresh_y=0.05, length_thresh=0.1
):
    line_len = abs(line_coords[1][0] - line_coords[0][0])
    box_len = abs(box_coords[1][0] - box_coords[0][0])

    # check if thay have size difference more than threshold
    if box_len > line_len:
        if box_len * (1 - length_thresh) >= line_len:
            return False
    else:
        if line_len * (1 - length_thresh) >= box_len:
            return False

    box_cmx, box_cmy = center_mass(box_coords)
    line_cmx, line_cmy = center_mass(line_coords)
    avg_x, avg_y = coords_avg2(box_coords, line_coords)
    delta_x = abs(box_cmx - line_cmx) / avg_x
    delta_y = abs(box_cmy - line_cmy) / avg_y

    # check if their center masses are further than threshold
    if delta_x >= thresh_x or delta_y >= (0.50 + thresh_y):
        return False

    return True


def check_overlap(
    dlib_coords,
    yolo_coords,
    thresh_x=0.05,
    thresh_y=0.05,
    length_thresh=0.1,
    height_thresh=0.1,
):
    yolo_len = abs(yolo_coords[1][0] - yolo_coords[0][0])
    dlib_len = abs(dlib_coords[1][0] - dlib_coords[0][0])
    yolo_height = abs(yolo_coords[1][1] - yolo_coords[0][1])
    dlib_height = abs(dlib_coords[1][1] - dlib_coords[0][1])

    # check if thay have size difference more than threshold
    if dlib_len > yolo_len:
        if dlib_len * (1 - length_thresh) >= yolo_len:
            print("len")
            return False
    else:
        if yolo_len * (1 - length_thresh) >= dlib_len:
            print("len")
            return False
    if dlib_height > yolo_height:
        if dlib_height * (1 - height_thresh) >= yolo_height:
            print("height")
            return False
    else:
        if yolo_height * (1 - height_thresh) >= dlib_height:
            print("height")
            return False

    dlib_cmx, dlib_cmy = center_mass(dlib_coords)
    yolo_cmx, yolo_cmy = center_mass(yolo_coords)
    avg_x, avg_y = coords_avg(dlib_coords, yolo_coords)
    delta_x = abs(dlib_cmx - yolo_cmx) / avg_x
    delta_y = abs(dlib_cmy - yolo_cmy) / avg_y

    # check if their center masses are further than threshold
    if delta_x >= thresh_x or delta_y >= (0.50 + thresh_y):
        print("center mass", delta_x >= thresh_x, delta_y >= (0.50 + thresh_y))
        return False

    return True


def landmarks2numpy(landmarks, len):
    res = np.zeros((len, 2), np.int)
    for n in range(len):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        res[n] = (x, y)
    return res
