from typing import AnyStr, Optional, Union, Callable
import os.path as osp
from pathlib import Path
from .detectors import DETECTORS
from .engine_backends import ENGINE_BACKENDS


scrfd_outputs = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']


def scrfd_onnx(model_path: Union[str, Path], backend: str = 'ONNXInfer', **kwargs):
    inference_backend = ENGINE_BACKENDS.get(backend)(model=model_path, output_order=scrfd_outputs, **kwargs)
    model = DETECTORS.get('SCRFD')(inference_backend=inference_backend)
    return model

