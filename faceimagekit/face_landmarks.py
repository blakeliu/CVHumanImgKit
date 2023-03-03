from typing import AnyStr, Optional, Union, Callable
import os.path as osp
from pathlib import Path
from .landmarks import LANDMARKERS
from .engine_backends import ENGINE_BACKENDS


def pfld_onnx(model_path: Union[str, Path], backend: str = 'ONNXInfer', **kwargs):
    inference_backend = ENGINE_BACKENDS.get(
        backend)(weight_file=model_path, **kwargs)
    model = LANDMARKERS.get('PFLD')(infer_backend=inference_backend)
    return model
