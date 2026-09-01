#!/usr/bin/env python3
"""Approach a red cube with the Panda gripper. Two PNGs + distance print."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

xml = os.path.expanduser(
    "~/inference/mujoco_menagerie/franka_emika_panda/panda-cube.xml"
)
out_dir = os.path.expanduser("~/inference/mujoco-out")
os.makedirs(out_dir, exist_ok=True)

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)

hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
if hand < 0 or cube < 0:
    raise SystemExit(f"missing body hand={hand} cube={cube}")


def dist_cm():
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(data.xpos[hand] - data.xpos[cube]) * 100)


def snap(name):
    path = os.path.join(out_dir, name)
    r = mujoco.Renderer(model, 480, 640)
    r.update_scene(data, camera=-1)
    Image.fromarray(r.render()).save(path)
    r.close()
    print("wrote", path)


# Default pose: arm up, cube on the floor in front.
mujoco.mj_resetData(model, data)
print("nq", model.nq, "hand", data.xpos[hand], "cube", data.xpos[cube])
print("distance_before_cm", round(dist_cm(), 1))
snap("cube-far.png")

# A reach pose (arm joints only; cube freejoint stays where it is).
# Values from Menagerie panda pickup, first 9 qpos = 7 arm + 2 fingers.
data.qpos[0:9] = np.array(
    [0.29, 0.50, -0.14, -2.18, -0.03, 2.52, -0.49, 0.04, 0.04]
)
print("distance_after_cm", round(dist_cm(), 1))
print("hand_after", data.xpos[hand], "cube", data.xpos[cube])
snap("cube-near.png")
print("OK")
