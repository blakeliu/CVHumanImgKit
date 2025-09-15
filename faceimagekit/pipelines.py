from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import cv2
from faceimagekit.utils import resize_image, rersize_points
from .face_detectors import scrfd_model
from .face_landmarks import rtmpose_model


class _BasePipeline(ABC):
    @abstractmethod
    def prepare(self, x):
        raise NotImplementedError()

    @abstractmethod
    def predict(x, *args, **kwargs):
        raise NotImplementedError()


EngineType = Union["ONNXInfer", "NCNNInfer", "OpencvInfer"]
DeviceType = Union["cpu", "gpu"]


class FaceLandmarkPipeline(_BasePipeline):
    ld_crop_height_ratio = 1.2
    ld_crop_wdith_ratio = 1.1

    def __init__(
        self,
        det_weight: str,
        landmark_weight: str,
        det_backend: EngineType,
        landmark_backend: EngineType,
        det_input_shape: tuple = (3, 640, 640),
        landmark_input_shape: tuple = (3, 256, 256),
        device: DeviceType = "cpu",
    ) -> None:
        """landmark检测pipeline: face detection -> face landmark

        Args:
            det_weight (str): detection weight file
            landmark_weight (str): landmark weight file
            det_backend (EngineType): detection backend
            landmark_backend (EngineType): landmark backend
            det_input_shape (tuple, optional): c h w. Defaults to (3, 640, 640).
            landmark_input_shape (tuple, optional): c h w. Defaults to (3, 256, 256).
            device (DeviceType, optional): 'cpu' or 'gpu'. Defaults to 'cpu'.
        """
        self.det_input_shape = det_input_shape
        self.landmark_input_shape = landmark_input_shape
        self._det_infer = scrfd_model(
            det_weight, det_backend, input_shape=det_input_shape
        )
        self._ld_infer = rtmpose_model(
            landmark_weight, landmark_backend, input_shape=landmark_input_shape
        )
        self.device = device

    def prepare(self):
        self._det_infer.prepare(device=self.device)
        self._ld_infer.prepare(device=self.device)

    def lds_infer(self, img, boxes: list = None):
        keypoints, scores = self._ld_infer.predict(img, boxes)
        results = []
        for kps in keypoints:
            results.append(
                {
                    "landmarks": kps[0],
                }
            )
        return results

    def crop_ld_face(self, img, bbox):
        height, width = img.shape[0:2]

        box = bbox[0:4]
        x1, y1, x2, y2 = box.astype(int)

        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size_w = int(max([w, h]) * self.ld_crop_wdith_ratio)
        size_h = int(max([w, h]) * self.ld_crop_height_ratio)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size_w // 2
        x2 = x1 + size_w
        y1 = cy - int(size_h // 2)
        y2 = y1 + size_h

        left = 0
        top = 0
        bottom = 0
        right = 0
        if x1 < 0:
            left = -x1
        if y1 < 0:
            top = -y1
        if x2 >= width:
            right = x2 - width
        if y2 >= height:
            bottom = y2 - height

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        cropped = img[y1:y2, x1:x2]

        cropped = cv2.copyMakeBorder(
            cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0
        )

        return {
            "face_img": cropped,
            "face_w": size_w,
            "face_h": size_h,
            "face_x_offset": x1 - left,
            "face_y_offset": y1 - top,
        }

    def predict(
        self, img: np.ndarray, score_threshold: float = 0.5, nms_threshold: float = 0.4
    ):
        res_img, scale_factor = resize_image(img, self.det_input_shape[::-1])
        height, width = img.shape[0:2]
        dets_list, kpss_list = self._det_infer.predict(
            res_img, score_threshold, nms_threshold
        )

        face_list = []
        for dets, kps in zip(dets_list[0], kpss_list[0]):
            bbox = rersize_points(dets[0:4], scale_factor)  # xyxy
            bbox = bbox.astype(np.int32)
            bbox[0::2] = np.clip(bbox[0::2], 0, width - 1)
            bbox[1::2] = np.clip(bbox[1::2], 0, height - 1)
            prob = dets[4]
            kps = rersize_points(kps, scale_factor)

            box_img_info = self.crop_ld_face(img, bbox)
            face_img = box_img_info["face_img"]
            lds_info = self.lds_infer(face_img)[0]
            lds_info["landmarks"][:, 0] += box_img_info["face_x_offset"]
            lds_info["landmarks"][:, 1] += box_img_info["face_y_offset"]
            lds_info["landmarks"] = lds_info["landmarks"].astype(np.int32)
            lds_info["landmarks"][:, 0] = np.clip(
                lds_info["landmarks"][:, 0], 0, width - 1
            )
            lds_info["landmarks"][:, 1] = np.clip(
                lds_info["landmarks"][:, 1], 0, height - 1
            )
            lds_info["bbox"] = bbox
            lds_info["prob"] = prob
            lds_info["kps"] = kps

            face_list.append(lds_info)
        return face_list
