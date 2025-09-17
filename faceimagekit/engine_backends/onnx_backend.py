import os.path as osp
import sys
import logging
import pkg_resources as pkg
import numpy as np
from faceimagekit.core import Registry, regsiter_fn, module_available
from faceimagekit.core.exception import ONNXRunException

if not module_available("onnxruntime"):
    raise ModuleNotFoundError(
        "onnxtuntime package not found! please 'pip install onnxtuntime'"
    )
import onnxruntime

if not module_available("GPUtil"):
    raise ModuleNotFoundError("gputil package not found! please 'pip install gputil'")
import GPUtil


def check_onnxruntime_gpu():
    cuda = True
    for i, r in enumerate("onnx", "onnxruntime-gpu"):
        try:
            pkg.require(r)
        except Exception:
            cuda = False
            print(f"{r} not found and is required by ONNXInfer")
    return cuda


class ONNXInfer:
    def __init__(self, weight_file, input_shape=None, output_order=None, **kwargs):
        self._model = None
        logging.info("ONNXInfer started")
        self.input = None
        self.input_dtype = None
        self.input_shape = input_shape  # c h w
        self.output_order = output_order
        self.out_shapes = None
        self._weight_file = weight_file
        if not osp.exists(self._weight_file):
            raise FileNotFoundError(f"onnx file: {self._weight_file} not found!")
        self.__dict__.update(**kwargs)

    def __del__(self):
        self._model = None

    # warmup
    def prepare(self, device: str = "cpu"):
        cuda = (
            "cuda" in device
            and check_onnxruntime_gpu()
            and len(GPUtil.getAvailable(order="first")) > 0
        )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        self._model = onnxruntime.InferenceSession(
            self._weight_file, providers=providers
        )
        meta = self._model.get_modelmeta().custom_metadata_map  # metadata
        self.input = self._model.get_inputs()[0]
        self.input_dtype = self.input.type
        if self.input_dtype == "tensor(float)":
            self.input_dtype = np.float32
        else:
            self.input_dtype = np.uint8
        if self.output_order is None:
            self.output_order = [e.name for e in self._model.get_outputs()]
        if "stride" in meta:
            stride, names = int(meta["stride"]), eval(meta["names"])
            self.stride = stride
            self.names = names
        logging.info("Warming up ONNX Runtime engine...")

        self.out_shapes = [e.shape for e in self._model.get_outputs()]

        self.input_shape = (self.input.shape[0], *self.input_shape)
        self.run(np.zeros(self.input_shape, self.input_dtype))
        # self._model.run(self.output_order,
        #                 {self.input.name: np.zeros(self.input_shape, self.input_dtype)})

    def run(self, input):
        try:
            net_out = self._model.run(self.output_order, {self.input.name: input})
        except Exception as e:
            raise ONNXRunException(f"onnx run error: {str(e)}")
        return net_out


def regsiter_onnx_backend(register: Registry):
    register(
        fn=ONNXInfer, name=ONNXInfer.__name__, namespace="engine_backend", type="onnx"
    )
