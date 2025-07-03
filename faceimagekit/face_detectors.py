from typing import AnyStr, Optional, Union, Callable
import os.path as osp
from pathlib import Path
from .detectors import DETECTORS
from .engine_backends import ENGINE_BACKENDS


scrfd_outputs = {
    "gnkps": {
        "500m": ["447", "512", "577", "450", "515", "580", "453", "518", "583"],
        "2.5g": ["448", "488", "528", "451", "491", "531", "454", "494", "534"],
        "10g": ["451", "504", "557", "454", "507", "560", "457", "510", "563"],
    },
    "bnkps": [
        "score_8",
        "score_16",
        "score_32",
        "bbox_8",
        "bbox_16",
        "bbox_32",
        "kps_8",
        "kps_16",
        "kps_32",
    ],
    "default": ["score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32"],
}


def get_scrfd_outputs(model_name: str):
    model_sizes = ["500m", "2.5g", "10g"]
    if "gnkps" in model_name:
        for m in model_sizes:
            if m in model_name:
                return scrfd_outputs["gnkps"][m], []
    elif "bnkps" in model_name:
        shape = []
        for m in model_sizes:
            if m in model_name:
                shape = [
                    int(s)
                    for s in model_name[len(f"scrfd_{m}_bnkps_shape") :].split("x")
                ]
        return scrfd_outputs["bnkps"], shape
    else:
        shape = []
        for m in model_sizes:
            if m in model_name:
                shape = [
                    int(s) for s in model_name[len(f"scrfd_{m}_shape") :].split("x")
                ]
        return scrfd_outputs["default"], shape


def scrfd_model(model_path: Union[str, Path], backend: str = "ONNXInfer", **kwargs):
    if backend == "ONNXInfer":
        model_name = osp.basename(model_path).split(".onnx")[0]
        scrfd_outputs_scale, shape = get_scrfd_outputs(model_name)
        input_shape = kwargs.pop("input_shape", [])
        if not input_shape:
            input_shape = [3, *shape]
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path,
            input_shape=input_shape,
            output_order=scrfd_outputs_scale,
            **kwargs,
        )
    elif backend == "NCNNInfer":
        model_name = osp.basename(model_path)
        scrfd_outputs_scale, shape = get_scrfd_outputs(model_name)
        input_shape = kwargs.pop("input_shape", [])
        if not input_shape:
            input_shape = [3, *shape]
        input_order = "input.1"
        inference_backend = ENGINE_BACKENDS.get(backend)(
            weight_file=model_path,
            input_shape=input_shape,
            input_order=input_order,
            output_order=scrfd_outputs_scale,
            **kwargs,
        )
    else:
        raise ValueError(
            f"backend must be 'ONNXInfer' or 'NCNNInfer', but got{str(backend)}"
        )
    model = DETECTORS.get("SCRFD")(infer_backend=inference_backend)
    return model
