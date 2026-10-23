"""ctypes binding for libt2.so (TRELLIS.2 image -> GLB in CUDA C++). Load only where a GPU is attached."""
import ctypes
import importlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("pipeline", "remesh", "simplify", "uv_unwrap", "bake", "export")


CUDA_LIBS = ("libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9")


def _preload_cuda_libs(lib_dirs=None):
    # libt2.so links the CUDA 12 runtime, cuBLAS and cuDNN 9 by soname: load them from lib_dirs, or else from the
    # nvidia-*-cu12 packages pip installed for a CUDA 12 torch.
    if lib_dirs is None:
        lib_dirs = []
        for mod in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cudnn"):
            try:
                lib_dirs.append(os.path.join(list(importlib.import_module(mod).__path__)[0], "lib"))
            except ImportError:
                pass
    for n in CUDA_LIBS:
        for d in lib_dirs:
            if os.path.exists(os.path.join(d, n)):
                ctypes.CDLL(os.path.join(d, n), mode=ctypes.RTLD_GLOBAL)
                break


class T2:
    def __init__(self, ckpt_dir, lib_path=os.path.join(HERE, "lib", "libt2.so"), fast=True, cuda_lib_dirs=None):
        os.environ["T2_FAST"] = "1" if fast else "0"
        # The GPU TELEA inpaint faults on sm_120 (fine on sm_90); use the exact CPU inpaint there.
        os.environ.setdefault("T2_TELEA_GPU", "0")
        os.environ.setdefault("T2_FLEXGEMM_CACHE", os.path.join(HERE, "assets", "flexgemm_autotune.json"))
        _preload_cuda_libs(cuda_lib_dirs)
        lib = ctypes.CDLL(lib_path)
        lib.t2_last_error.restype = ctypes.c_char_p
        lib.t2_load.argtypes = [ctypes.c_char_p]
        lib.t2_image_to_glb.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_int64)]
        self.lib = lib
        self._check(lib.t2_load(ckpt_dir.encode()))

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError(self.lib.t2_last_error().decode(errors="replace"))

    def image_to_glb(self, rgba: np.ndarray, out_glb: str, seed=42, faces=1_000_000, texture=4096):
        """rgba: uint8 [H, W, 4] with a real alpha channel. Returns (stage seconds dict, vertices, faces)."""
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        assert rgba.ndim == 3 and rgba.shape[2] == 4
        t = (ctypes.c_double * len(STAGES))()
        nv, nf = ctypes.c_int64(), ctypes.c_int64()
        self._check(self.lib.t2_image_to_glb(rgba.ctypes.data, rgba.shape[1], rgba.shape[0], seed, faces, texture, out_glb.encode(),
                                             t, ctypes.byref(nv), ctypes.byref(nf)))
        return dict(zip(STAGES, list(t))), nv.value, nf.value
