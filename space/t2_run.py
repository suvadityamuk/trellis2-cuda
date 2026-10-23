"""One image -> GLB with libt2.so in its own process; prints a JSON result as the last stdout line.

The app's process holds torch's CUDA 13 libraries, whose cuDNN shares the libcudnn.so.9 soname with the CUDA 12 cuDNN that
libt2.so needs, so the port runs here with only its own CUDA 12 libraries loaded. Usage: python t2_run.py '<json args>'"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from t2 import T2

a = json.loads(sys.argv[1])
try:
    t0 = time.perf_counter()
    m = T2(a["ckpt"], fast=True, cuda_lib_dirs=a["lib_dirs"])
    load = time.perf_counter() - t0
    rgba = np.asarray(Image.open(a["image"]).convert("RGBA"))
    times, nv, nf = m.image_to_glb(rgba, a["out"], seed=a["seed"], faces=a["faces"], texture=a["texture"])
    print(json.dumps({"load": load, "times": times, "vertices": nv, "faces": nf, "fast": os.environ["T2_FAST"] == "1"}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
