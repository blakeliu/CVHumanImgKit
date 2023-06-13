from typing import AnyStr, Optional, Union, Callable
import os.path as osp
from pathlib import Path
from .segments import SEGMENTERS
from .engine_backends import ENGINE_BACKENDS


pplitesegface_outputs = ['519']
pplitesegface12_outputs = ['741']


def pplitesegface_model(model_path: Union[str, Path], backend: str = 'ONNXInfer', **kwargs):
    if backend == 'ONNXInfer':
        inference_backend = ENGINE_BACKENDS.get(
            backend)(weight_file=model_path, **kwargs)
    elif backend == 'NCNNInfer':
        input_shape = kwargs.pop("input_shape", (3, 512, 512))
        input_order = "input.1"
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path, input_shape=input_shape, input_order=input_order, output_order=pplitesegface_outputs, **kwargs)
    model = SEGMENTERS.get('PPLiteSeg')(infer_backend=inference_backend)
    return model


def pplitesegface12_model(model_path: Union[str, Path], backend: str = 'ONNXInfer', **kwargs):
    if backend == 'ONNXInfer':
        inference_backend = ENGINE_BACKENDS.get(
            backend)(weight_file=model_path, **kwargs)
    elif backend == 'NCNNInfer':
        input_shape = kwargs.pop("input_shape", (3, 512, 512))
        input_order = "input.1"
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path, input_shape=input_shape, input_order=input_order, output_order=pplitesegface12_outputs, **kwargs)
    model = SEGMENTERS.get('PPLiteSeg')(infer_backend=inference_backend)
    return model
