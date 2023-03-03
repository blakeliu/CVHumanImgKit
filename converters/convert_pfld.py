import os
from pathlib import Path
import logging
import numpy as np
import torch
import onnxruntime
from flashdl.utilities.attr import ImageDevice, ImageSource, AttrMedFace, ImageSourceNamePair
from app.infer import LandmarkInfer
from flashdl.utilities.image.opencv import cv_show, cv_keyboard_run

project_name = "TFace_Atrributes"
cur_path = os.path.abspath('.')
parent_dir = cur_path.split(project_name)[0]
project_dir = Path(parent_dir).joinpath(project_name)

data_dir = "/home/tf/data/disk/data/face/3dpro/images/org"
cfg_file = project_dir.joinpath("task/attr/threedimpro/Necklines/toml/resnet_v1.0.0.toml")

det_param = {"model": "resnet50", "device": "cpu",
                "model_path": str(project_dir.joinpath("model_zoom/face_detection/Resnet50_Final.pth")),
                "resize": 0.25}
landmark_param = {"model": "PFLDInference", "device": "cpu",
                    "model_path": str(project_dir.joinpath("model_zoom/face_landmark/landmark_lapa_134.pth"))}
logger_file = "debug.txt"

ld = LandmarkInfer(det_param, landmark_param, logger_file=logger_file, level=logging.DEBUG)
ld.warm_up()

torch_img = torch.zeros(1, 3, 112, 112).to(torch.device('cpu'))
with torch.no_grad():
    torch_out = ld.session(torch_img)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_file = "landmark_lapa_134.onnx"

torch.onnx.export(ld.session, torch_img, onnx_file, verbose=False, opset_version=12, output_names=['pose', 'lds'])

ort_session = onnxruntime.InferenceSession(onnx_file)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch_img)}
ort_outs = ort_session.run(None, ort_inputs)

np.testing.assert_allclose(
    to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)


