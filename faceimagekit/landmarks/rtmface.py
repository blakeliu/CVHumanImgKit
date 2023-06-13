import os
import sys
import logging
import pkg_resources as pkg
import numpy as np
from faceimagekit.core import Registry, regsiter_fn, module_available
if not module_available("numba"):
    raise ModuleNotFoundError(
        "numba package not found! please 'pip install numba'")
from .base import Landmarker

class RTMFace(Landmarker):
    def __init__(self, infer_backend, version=1) -> None:
        """
        https://e.coding.net/blakeliu/face/mmpose.git
        tools/training_lab.ipynb
        configs/face_2d_keypoint/rtmpose/lapa/rtmpose-m_8xb64-120e_lapa134-256x256.py
        configs/face_2d_keypoint/rtmpose/lapa/rtmpose-s_8xb64-120e_lapa134-256x256.py
        Args:
            infer_backend (_type_): _description_
            version (int, optional): _description_. Defaults to 1.
        """
        super().__init__(infer_backend, version)
        self.version = version
        self.input_shape = (256, 256, 3)
        self.out_shapes = None
        self.infer_shape = None
        
    def prepare(self, **kwargs):
        """
        Read network params and populate class paramters.
        Args:
            nms_threshold (float, optional): _description_. Defaults to 0.4.
        """
        self.session.prepare(**kwargs)
        self.infer_shape = self.input_shape
        
    
    def predict(self, img, bboxes:list[np.ndarray]=None):
        """预测lds

        Args:
            img (_type_): 单张图像
            bboxes (list[np.ndarray], optional): [[x1,y1,x2,y2],...]. Defaults to None.
        """
        assert img is not None
        if bboxes is not None and len(bboxes) > 1:
            result = self.session.run(img, bboxes)
        else:
            result = self.session.run(img)
        return result
        
        
def regsiter_rtmface_landmarks(register: Registry):
    register(fn=RTMFace, name=RTMFace.__name__,
             namespace="landmarks", type="face_landmark")