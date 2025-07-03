from faceimagekit.core import Registry
from .pfld import regsiter_pfld_landmarks
from .rtmface import regsiter_rtmface_landmarks
from .rtmpose import regsiter_rtmpose_landmarks

LANDMARKERS = Registry("landmarkers")
regsiter_pfld_landmarks(LANDMARKERS)
regsiter_rtmface_landmarks(LANDMARKERS)
regsiter_rtmpose_landmarks(LANDMARKERS)