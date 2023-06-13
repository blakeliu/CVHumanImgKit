import os.path as osp
import sys
import logging
import cv2
import numpy as np
from faceimagekit.core import Registry, regsiter_fn, module_available
if not module_available("mmdeploy_runtime"):
    raise ModuleNotFoundError(
        "mmdeploy_runtime package not found! please 'pip install mmdeploy_runtime'")
from mmdeploy_runtime import PoseDetector


class MMDeployLoadError(Exception):
    pass


class MMDeployRunError(Exception):
    pass


class MMDeployInfer:
    def __init__(self, weight_file,
                 input_shape=None,
                 output_order=None, **kwargs):

        self._model = None
        logging.info('MMDeployInfer started')
        self.input = None
        self.input_dtype = None
        self.input_shape = input_shape  # c h w
        self.output_order = output_order
        self.out_shapes = None
        self._weight_file = weight_file
        if not osp.exists(self._weight_file):
            raise FileNotFoundError(
                f"mmdeploy weight dir: {self._weight_file} not found!")
        self.__dict__.update(**kwargs)

    def __del__(self):
        self._model = None

    # warmup
    def prepare(self, device: str = 'cpu'):
        try:
            self._model = PoseDetector(
                model_path=self._weight_file, device_name=device, device_id=0)
        except Exception as e:
            raise MMDeployLoadError(f"PoseDetector init failed: {str(e)}")
        self.input_dtype = np.uint8
        logging.info("Warming up mmdeploy runtime engine...")
        self.run(np.zeros(self.input_shape, self.input_dtype))

    def run(self, input, bboxes: list = None):
        try:
            if bboxes is not None and len(bboxes) > 1:
                result = self._model(input, bboxes)
            else:
                result = self._model(input)
        except Exception as e:
            raise MMDeployRunError(f"mmdeploy run error: {str(e)}")
        return result


def regsiter_mmdeploy_backend(register: Registry):
    register(fn=MMDeployInfer, name=MMDeployInfer.__name__,
             namespace="engine_backend", type="onnx")
