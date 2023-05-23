import os.path as osp
import sys
from typing import Dict
import logging
import pkg_resources as pkg
import numpy as np
from faceimagekit.core import Registry, regsiter_fn, module_available
from faceimagekit.core.exception import NCNNRunException
if not module_available("ncnn"):
    raise ModuleNotFoundError(
        "ncnn package not found! please 'pip install ncnn'")
import ncnn


class NCNNInfer:
    def __init__(self, weight_file,
                 input_shape=None,
                 input_order=None,
                 output_order=None, **kwargs):

        self._model = None
        logging.info('NCNNInfer started')
        self.input_order = input_order
        self.input_shape = input_shape  # c h w
        self.output_order = output_order
        self.input_dtype = None
        self.out_shapes = None
        self._param_file = weight_file+".param"
        self._bin_file = weight_file+".bin"
        if not osp.exists(self._param_file):
            raise FileNotFoundError(
                f"param file: {self._param_file} not found!")
        if not osp.exists(self._bin_file):
            raise FileNotFoundError(f"bin file: {self._bin_file} not found!")
        self.num_threads = kwargs.pop("num_threads", 4)
        self.__dict__.update(**kwargs)

    # warmup
    def prepare(self, device: str = 'cpu'):
        self._model = ncnn.Net()
        self._model.opt.num_threads = self.num_threads
        if device == "gpu":
            self._model.opt.use_vulkan_compute = True
        else:
            self._model.opt.use_vulkan_compute = False
        self._model.load_param(self._param_file)
        self._model.load_model(self._bin_file)
        c, h, w = self.input_shape
        in_mat = ncnn.Mat((w, h, c))
        self.input_shape = (1, *self.input_shape)
        out_puts = self.run(in_mat)

    def pre_process(self, input):
        if isinstance(input, ncnn.Mat):
            in_mat = input
        elif isinstance(input, np.ndarray):
            # np convert ncnn
            in_mat = input[0]  # c h w
            np_mat = np.ascontiguousarray(in_mat)
            in_mat = ncnn.Mat(np_mat)  # w,h,c
            raise NotImplementedError(f"np ndarray convert to ncnn:Mat may be failed!, please input as ncnn::Mat!")
        else:
            raise ValueError(
                f"input type must be ncnn.Mat or np.ndarray, but got: {type(input)}")
        return in_mat

    def run(self, input):
        in_mat = self.pre_process(input)
        out_puts = {}
        try:
            with self._model.create_extractor() as ex:
                if isinstance(self.input_order, list) or isinstance(self.input_order, tuple):
                    for in_name in self.input_order:
                        ex.input(in_name, in_mat)
                else:
                    ex.input(self.input_order, in_mat)

                for out_name in self.output_order:
                    ret, out_puts[out_name] = ex.extract(out_name)
        except Exception as e:
            raise NCNNRunException(f"ncnn run error: {str(e)}")
        return self.post_process(out_puts)

    def post_process(self, out_puts: Dict):
        np_outputs = []
        for k, v in out_puts.items():
            np_outputs.append(np.expand_dims(np.array(v, np.float32), 0))
        return np_outputs

    def __del__(self):
        self._model = None


def regsiter_ncnn_backend(register: Registry):
    register(fn=NCNNInfer, name=NCNNInfer.__name__,
             namespace="engine_backend", type="ncnn")
