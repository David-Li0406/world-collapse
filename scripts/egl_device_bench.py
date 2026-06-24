"""Benchmark MuJoCo EGL rendering PER EGL device + isolate physics vs render.

Goal: determine whether the post-reset slowdown is (a) per-GPU render
contention / wrong EGL-device binding (some device renders fast), or (b) a
global render regression (all devices slow), or (c) physics not render.

Prints, for each EGL device id: GL_RENDERER (which physical GPU) and ms/render;
plus a physics-vs-render split on the push-v3 env.
"""
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import ctypes
import glob
for _p in sorted(glob.glob("/opt/nvidia-*/lib64/libEGL_nvidia.so.0")):
    try:
        ctypes.cdll.LoadLibrary(_p)
        print(f"[bench] preloaded NVIDIA EGL ICD: {_p}", flush=True)
        break
    except OSError:
        pass

import numpy as np
import mujoco
print(f"[bench] mujoco {mujoco.__version__}", flush=True)

# Enumerate EGL devices via eglQueryDevicesEXT.
try:
    from OpenGL import EGL
    import OpenGL.EGL as egl
    egl.eglQueryDevicesEXT  # may not be wrapped; fall through to ctypes if missing
    n = EGL.EGLint(0)
    maxd = 16
    devs = (ctypes.c_void_p * maxd)()
    # use eglGetProcAddress
    _egl = ctypes.cdll.LoadLibrary("libEGL.so.1")
    _egl.eglGetProcAddress.restype = ctypes.c_void_p
    QD = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
                          ctypes.POINTER(ctypes.c_void_p),
                          ctypes.POINTER(ctypes.c_int))(
        _egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
    cnt = ctypes.c_int(0)
    QD(maxd, devs, ctypes.byref(cnt))
    print(f"[bench] EGL devices enumerated: {cnt.value}", flush=True)
    NDEV = max(1, cnt.value)
except Exception as e:
    print(f"[bench] EGL enumerate failed: {e!r}; will probe ids 0..7", flush=True)
    NDEV = 8

XML = """
<mujoco><worldbody>
  <light pos='0 0 2'/>
  <geom type='plane' size='1 1 .1'/>
  <body pos='0 0 .3'><freejoint/><geom type='box' size='.1 .1 .1' rgba='.8 .3 .3 1'/></body>
</worldbody></mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

from OpenGL import GL
for dev in range(min(NDEV, 8)):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
    try:
        r = mujoco.Renderer(model, 64, 64)
        renderer_str = GL.glGetString(GL.GL_RENDERER)
        for _ in range(3):  # warmup
            r.update_scene(data); r.render()
        t = time.time()
        N = 30
        for _ in range(N):
            r.update_scene(data); r.render()
        ms = (time.time() - t) / N * 1000
        print(f"[bench] EGL_DEVICE_ID={dev}  GL_RENDERER={renderer_str}  render={ms:.1f} ms/frame", flush=True)
        r.close()
    except Exception as e:
        print(f"[bench] EGL_DEVICE_ID={dev}  FAILED: {e!r}", flush=True)
