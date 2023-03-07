from typing import AnyStr, Optional, Union, Callable
import os.path as osp
from pathlib import Path
from .detectors import DETECTORS
from .engine_backends import ENGINE_BACKENDS


scrfd_outputs = ['score_8', 'score_16', 'score_32', 'bbox_8',
                 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
scrfd_10g_outputs_scale = ['451', '504', '557',
                           '454', '507', '560', '457', '510', '563']
scrfd_2dot5g_outputs_scale = ['448', '488',
                              '528', '451', '491', '531', '454', '494', '534']
scrfd_500m_outputs_scale = ['447', '512', '577',
                            '450', '515', '580', '453', '518', '583']


def scrfd_model(model_path: Union[str, Path], backend: str = 'ONNXInfer', **kwargs):
    if backend == 'ONNXInfer':
        model_name = osp.basename(model_path).split(".onnx")[0]
        if model_name == "scrfd_500m_gnkps":
            scrfd_outputs_scale = scrfd_500m_outputs_scale
        elif model_name == "scrfd_2.5g_gnkps":
            scrfd_outputs_scale = scrfd_2dot5g_outputs_scale
        elif model_name == "scrfd_10g_gnkps":
            scrfd_outputs_scale = scrfd_10g_outputs_scale
        else:
            raise ValueError(f"Unknow model file: {model_path}")
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path, output_order=scrfd_outputs_scale, **kwargs)
    elif backend == 'NCNNInfer':
        input_shape = kwargs.pop("input_shape", (3, 640, 640))
        input_order = "input.1"
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path, input_shape=input_shape, input_order=input_order, output_order=scrfd_outputs_scale, **kwargs)
    else:
        raise ValueError(
            f"backend must be 'ONNXInfer' or 'NCNNInfer', but got{str(backend)}")
    model = DETECTORS.get('SCRFD')(infer_backend=inference_backend)
    return model
