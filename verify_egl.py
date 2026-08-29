#!/usr/bin/env python3
"""Headless EGL smoke test: step a bundled model, write one PNG."""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image

print("mujoco", mujoco.__version__, "MUJOCO_GL=", os.environ.get("MUJOCO_GL"))
xml = os.path.join(os.path.dirname(mujoco.__file__), "testdata", "model.xml")
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
t0 = time.perf_counter()
for _ in range(2000):
    mujoco.mj_step(model, data)
elapsed = time.perf_counter() - t0
sps = 2000 / elapsed if elapsed else 0
print("steps_per_sec", round(sps, 1))
out = os.path.expanduser("~/inference/mujoco-out/egl-smoke.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data)
Image.fromarray(renderer.render()).save(out)
renderer.close()
print("wrote", out, os.path.getsize(out), "bytes")
if sps < 100 or os.path.getsize(out) < 1000:
    sys.exit(1)
print("OK")
