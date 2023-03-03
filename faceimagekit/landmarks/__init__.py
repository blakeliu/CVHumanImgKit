from faceimagekit.core import Registry
from .pfld import regsiter_pfld_landmarks

LANDMARKERS = Registry("landmarkers")
regsiter_pfld_landmarks(LANDMARKERS)