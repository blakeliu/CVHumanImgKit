import os
import sys
import logging
import pkg_resources as pkg
import numpy as np
from faceimagekit.core import Registry, regsiter_fn, module_available
if not module_available("numba"):
    raise ModuleNotFoundError(
        "numba package not found! please 'pip install numba'")
from numba import njit
import cv2
from faceimagekit.utils import rersize_points, resize_image

from .base import Landmarker


def normalize_on_np(input):
    img = np.asarray(input, dtype=np.float32)
    img = img[..., ::-1]
    img = np.transpose(img, (0, 3, 1, 2))
    img = np.multiply(img, 1/255.0)
    return img

class PFLD(Landmarker):
    def __init__(self, infer_backend, version=1) -> None:
        """
        PFLD: A Practical Facial Landmark Detector.2019
        参考:
        https://github.com/tfrbt/TFace_Atrributes/master/app/infer/plfd_infer.py
        Args:
            infer_backend (): _description_
            version (int, optional): _description_. Defaults to 1.
        """
        super().__init__(infer_backend, version)
        self.session = infer_backend
        self.center_cache = {}
        self.nms_threshold = 0.4
        self.masks = False
        self.version = version
        self.input_shape = (1, 3, 122, 122)
        self.out_shapes = None
        self.infer_shape = None
        self.score_list = None
        self.kps_list = None


    def prepare(self, nms_threshold: float = 0.4, **kwargs):
        """
        Read network params and populate class paramters.
        Args:
            nms_threshold (float, optional): _description_. Defaults to 0.4.
        """
        self.nms_threshold = nms_threshold
        self.session.prepare(**kwargs)
        self.out_shapes = self.session.out_shapes
        self.input_shape = self.session.input_shape
        self.infer_shape = self.input_shape

    def resize_img(self, img):
        return resize_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.input_shape[2:][::-1])

    def detect(self, imgs, resize_shapes:list=None):
        """_summary_

        Args:
            imgs (_type_): _description_
            resize_shapes (list, optional): h, w

        Returns:
            _type_: _description_
        """
        res_imgs = []
        scale_imgs = []
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            for img in imgs:
                res_img, scale = self.resize_img(img)
                res_imgs.append(res_img)
                scale_imgs.append(scale)
        elif isinstance(imgs, np.ndarray):
            res_img, scale = self.resize_img(imgs)
            res_imgs.append(res_img)
            scale_imgs.append(scale)
        

        if len(res_imgs) == 1:
            res_imgs = np.expand_dims(res_imgs[0], 0)
        else:
            res_imgs = np.stack(res_imgs)

        blobs = self._preprocess(res_imgs)
        return self._postprocess(self._forward(blobs), scale_imgs, resize_shapes)

    def _preprocess(self, img: np.ndarray):
        blob = normalize_on_np(img)
        return blob
    
    def _postprocess(self, net_outputs, scale_imgs, resize_shapes):
        poses, landamrks = net_outputs
        poses_list = []
        landmarks_list = []
        for i, lds in enumerate(landamrks):
            poses_list.append(poses[i])
            if resize_shapes is not None and len(resize_shapes) > 0:
                landmarks_list.append(rersize_points(lds.reshape(-1, 2)*resize_shapes[i][::-1], scale_imgs[i]))
            else:
                landmarks_list.append(rersize_points(lds.reshape(-1, 2)*self.input_shape[2:][::-1], scale_imgs[i]))
        return poses_list, landmarks_list

    def _forward(self, blob):
        """
        send input data to infer backend
        Args:
            blob (_type_): _description_
        """
        assert blob is not None
        output = self.session.run(blob)
        return output


def regsiter_pfld_landmarks(register: Registry):
    register(fn=PFLD, name=PFLD.__name__,
             namespace="landmarks", type="face_landmark")
