#!/usr/bin/env python3
"""Panda joint2 (shoulder) +40 deg, before/after PNG.

From the default side camera, joint1 yaw barely shows; joint2 does.
"""
import os
import math

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image

xml = os.path.expanduser("~/inference/mujoco_menagerie/franka_emika_panda/scene.xml")
out_dir = os.path.expanduser("~/inference/mujoco-out")
os.makedirs(out_dir, exist_ok=True)

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
if joint_id < 0:
    raise SystemExit("joint2 not found")
adr = int(model.jnt_qposadr[joint_id])
print("joint joint2 qpos_index", adr, "before_rad", float(data.qpos[adr]))


def snap(name):
    path = os.path.join(out_dir, name)
    r = mujoco.Renderer(model, 480, 640)
    r.update_scene(data)
    Image.fromarray(r.render()).save(path)
    r.close()
    print("wrote", path, os.path.getsize(path), "bytes")


snap("joint2-before.png")

data.qpos[adr] = float(data.qpos[adr]) + math.radians(40)
mujoco.mj_forward(model, data)
print("after_rad", float(data.qpos[adr]))
snap("joint2-after.png")
print("OK")
