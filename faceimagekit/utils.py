from typing import Dict, List
import time
import numpy as np
import cv2


def draw_face(image: np.ndarray, faces: List[Dict[str, np.ndarray]], draw_socre: bool = False, draw_lanamrk:bool=False):
    for face in faces:
        box = face["bbox"].astype(int)
        pt1 = tuple(box[0:2])
        pt2 = tuple(box[2:4])
        x, y = pt1
        r, b = pt2
        w = r - x
        color = (0, 255, 0)
        cv2.rectangle(image, pt1, pt2, color, 1)

        if draw_socre:
            text = f"{face['prob']:.3f}"
            pos = (x + 3, y - 5)
            textcolor = (0, 0, 0)
            thickness = 1
            border = int(thickness / 2)
            cv2.rectangle(image, (x - border, y - 21, w +
                          thickness, 21), color, -1, 16)
            cv2.putText(image, text, pos, 0, 0.5, color, 3, 16)
            cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

    return image


def rersize_points(dets, scale: float):
    if scale != 1.0:
        dets = dets / scale
    return dets


def resize_image(image, max_size: list = None):
    if max_size is None:
        max_size = [640, 640]

    cw = max_size[0]
    ch = max_size[1]
    h, w, _ = image.shape

    scale_factor = min(cw / w, ch / h)
    # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
    # so we reduce scale factor by some factor
    if scale_factor > 2:
        scale_factor = scale_factor * 0.7

    if scale_factor <= 1.:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    if scale_factor == 1.:
        transformed_image = image
    else:
        transformed_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                       fy=scale_factor,
                                       interpolation=interp)

    h, w, _ = transformed_image.shape

    if w < cw:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, 0, 0, cw - w,
                                               cv2.BORDER_CONSTANT)
    if h < ch:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, ch - h, 0, 0,
                                               cv2.BORDER_CONSTANT)

    return transformed_image, scale_factor


class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def time(self):
        return time.time() - self.start_time
