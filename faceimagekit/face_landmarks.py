from typing import Union
from pathlib import Path
from .landmarks import LANDMARKERS
from .engine_backends import ENGINE_BACKENDS


pfld_outputs = ["pose", "lds"]


def pfld_model(model_path: Union[str, Path], backend: str = "ONNXInfer", **kwargs):
    if backend == "ONNXInfer":
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path, **kwargs
        )
    elif backend == "NCNNInfer":
        input_shape = kwargs.pop("input_shape", (3, 112, 112))
        input_order = "input.1"
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path,
            input_shape=input_shape,
            input_order=input_order,
            output_order=pfld_outputs,
            **kwargs,
        )
    model = LANDMARKERS.get("PFLD")(infer_backend=inference_backend)
    return model


def rtmpose_model(model_path: Union[str, Path], backend: str, **kwargs):
    inference_backend = ENGINE_BACKENDS.get(backend)(weight_file=model_path, **kwargs)
    model = LANDMARKERS.get("RTMPose")(infer_backend=inference_backend)
    return model
