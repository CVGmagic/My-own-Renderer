import numpy as np
import cv2

golden_gate = "HDRIs/golden_gate_hills_4k.hdr"
hdri_bgr = cv2.imread(golden_gate)

if hdri_bgr is None:
    raise FileNotFoundError("Could not open or find the HDR image.")
# OpenCV loads images in BGR order. Convert to RGB for your raytracer.
default_hdri = cv2.cvtColor(hdri_bgr, cv2.COLOR_BGR2RGB) / 255
