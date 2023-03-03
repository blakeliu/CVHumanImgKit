from typing import AnyStr, Optional, Union, Callable
import os.path as osp
from pathlib import Path
from .detectors import DETECTORS
from .engine_backends import ENGINE_BACKENDS


scrfd_outputs = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
# scrfd_outputs = ['447', '512', '577', '450', '515', '580', '453', '518', '583']


def scrfd_onnx(model_path: Union[str, Path], backend: str = 'ONNXInfer', **kwargs):
    inference_backend = ENGINE_BACKENDS.get(backend)(weight_file=model_path, output_order=scrfd_outputs, **kwargs)
    model = DETECTORS.get('SCRFD')(infer_backend=inference_backend)
    return model

