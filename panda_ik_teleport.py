#!/usr/bin/env python3
"""Pitfall reproduction: snap IK qpos so the fingertip sits at the cube centre.

mj_forward does not run contacts. The cube occupies the same volume as the
hand. A later close+lift can report GRASP_OK while the PNG still shows the
cube through the palm. That is not a grasp.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mink
import mujoco
import numpy as np
from PIL import Image
from mink import SE3

robot_xml = os.path.expanduser("~/inference/mujoco_menagerie/franka_emika_panda/scene.xml")
world_xml = os.path.expanduser(
    "~/inference/mujoco_menagerie/franka_emika_panda/seance-cube.xml"
)
out_dir = os.path.expanduser("~/inference/mujoco-out")
os.makedirs(out_dir, exist_ok=True)

robot = mujoco.MjModel.from_xml_path(robot_xml)
world = mujoco.MjModel.from_xml_path(world_xml)
wdata = mujoco.MjData(world)
lf = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
cube = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "cube")

configuration = mink.Configuration(robot)
configuration.update_from_keyframe("home")
tip = mink.FrameTask("left_finger", "body", position_cost=1.0, orientation_cost=0.0)
posture = mink.PostureTask(robot, cost=1e-3)
posture.set_target_from_configuration(configuration)
tasks = [tip, posture]


def snap(name):
    path = os.path.join(out_dir, name)
    r = mujoco.Renderer(world, 480, 640)
    r.update_scene(wdata)
    Image.fromarray(r.render()).save(path)
    r.close()
    print("wrote", path)


mujoco.mj_resetData(world, wdata)
wdata.qpos[0:9] = configuration.q[:9]
mujoco.mj_forward(world, wdata)
cube_pos = wdata.xpos[cube].copy()
target = cube_pos + np.array([0.0, 0.0, 0.01])
tip.set_target(SE3.from_translation(target))
for _ in range(800):
    vel = mink.solve_ik(configuration, tasks, 0.01, "daqp", damping=1e-3)
    configuration.integrate_inplace(vel, 0.01)
wdata.qpos[0:9] = configuration.q[:9]
mujoco.mj_forward(world, wdata)
print("teleport fingers_to_cube_cm", round(float(np.linalg.norm(wdata.xpos[lf] - wdata.xpos[cube]) * 100), 1))
snap("pitfall-teleport.png")
print("OK")
