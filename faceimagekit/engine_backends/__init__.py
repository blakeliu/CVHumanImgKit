from faceimagekit.core import Registry
from .onnx_backend import regsiter_onnx_backend

ENGINE_BACKENDS = Registry("engine_backends")
regsiter_onnx_backend(ENGINE_BACKENDS)