"""Diagnose the post-reset slowdown in the main (iVideoGPT/DrQ-v2) env.

Reports: torch CUDA availability + GPU vs CPU matmul speed, key package versions
(flagging anything MISSING), and push-v3 env per-step physics+render timing
(slow => software-render fallback). Run via the main uv env on the runner.
"""
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

# Preload NVIDIA EGL ICD (same as online_drqv2 / build_probe_bank).
import ctypes
import glob
for _p in sorted(glob.glob("/opt/nvidia-*/lib64/libEGL_nvidia.so.0")):
    try:
        ctypes.cdll.LoadLibrary(_p)
        print(f"[diag] preloaded NVIDIA EGL ICD: {_p}", flush=True)
        break
    except OSError as e:
        print(f"[diag] preload failed {_p}: {e}", flush=True)

import numpy as np
import torch

print(f"[diag] python torch={torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"torch.version.cuda={torch.version.cuda}  n_devices={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"[diag] device0={torch.cuda.get_device_name(0)}", flush=True)

# GPU vs CPU compute speed (a CPU-only env shows up here). Warm up first so the
# GPU number isn't dominated by one-time CUDA/cuBLAS init.
for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
    x = torch.randn(2048, 2048, device=dev)
    for _ in range(3):  # warmup
        y = x @ x
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        y = x @ x
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"[diag] matmul 2048^3 x10 on {dev}: {(time.time()-t)*1000:.0f} ms", flush=True)

# Key package versions (flag missing).
import importlib.metadata as M
for pkg in ["torch", "torchvision", "mujoco", "transformers", "diffusers",
            "numpy", "accelerate", "gymnasium", "metaworld", "h5py", "lpips",
            "einops", "safetensors", "hydra-core", "omegaconf"]:
    try:
        print(f"[diag] pkg {pkg}=={M.version(pkg)}", flush=True)
    except Exception:
        print(f"[diag] pkg {pkg}: MISSING", flush=True)

import mujoco
print(f"[diag] mujoco runtime={mujoco.__version__}", flush=True)

# Which GL device does MuJoCo's EGL context actually render on?
# 'llvmpipe'/'softpipe'/'Mesa' => SOFTWARE fallback (the slowdown); 'NVIDIA' => hardware.
try:
    _ctx = mujoco.GLContext(64, 64)
    _ctx.make_current()
    from OpenGL import GL
    print(f"[diag] GL_VENDOR={GL.glGetString(GL.GL_VENDOR)}", flush=True)
    print(f"[diag] GL_RENDERER={GL.glGetString(GL.GL_RENDERER)}", flush=True)
    print(f"[diag] GL_VERSION={GL.glGetString(GL.GL_VERSION)}", flush=True)
except Exception as e:
    print(f"[diag] GL renderer query failed: {e!r}", flush=True)

# push-v3 env per-step (physics + render) timing — slow => software render.
try:
    from wcollapse.envs.metaworld_dmenv import make
    env = make("push-v3", frame_stack=3, action_repeat=2, seed=0,
               camera="corner", duration=100, succ_bonus=10.0)
    t = time.time()
    env.reset()
    print(f"[diag] env.reset+first-render: {(time.time()-t):.2f}s", flush=True)
    a = np.zeros(env.action_spec().shape, np.float32)
    t = time.time()
    N = 8
    for _ in range(N):
        env.step(a)
    print(f"[diag] env.step (phys+render) mean: {(time.time()-t)/N*1000:.1f} ms/step "
          f"({N} steps)", flush=True)
    print("[diag] (healthy ~10-40 ms/step on NVIDIA EGL; ~seconds => software fallback)",
          flush=True)
except Exception as ex:
    import traceback
    traceback.print_exc()
    print(f"[diag] env timing FAILED: {ex!r}", flush=True)
