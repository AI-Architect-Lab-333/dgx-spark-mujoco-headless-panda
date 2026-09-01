#!/usr/bin/env python3
"""Pitfall reproduction: snap IK qpos so the fingertip sits at the cube centre.

mj_forward does not run contacts. The cube occupies the same volume as the
hand. Close+lift then prints GRASP_OK while the PNG still shows the cube
through the palm. That print is the pitfall, not a grasp.
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
    "~/inference/mujoco_menagerie/franka_emika_panda/panda-cube.xml"
)
out_dir = os.path.expanduser("~/inference/mujoco-out")
os.makedirs(out_dir, exist_ok=True)

robot = mujoco.MjModel.from_xml_path(robot_xml)
world = mujoco.MjModel.from_xml_path(world_xml)
wdata = mujoco.MjData(world)
lf = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
cube = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "cube")

configuration = mink.Configuration(robot)
configuration.update_from_keyframe("home")
tip = mink.FrameTask(
    "left_finger",
    "body",
    position_cost=1.0,
    orientation_cost=0.0,
    lm_damping=1e-2,
)
posture = mink.PostureTask(robot, cost=1e-3)
posture.set_target_from_configuration(configuration)
tasks = [tip, posture]
dt = 0.01


def apply_q():
    wdata.qpos[0:9] = configuration.q[:9]
    wdata.ctrl[0:7] = configuration.q[:7]
    mujoco.mj_forward(world, wdata)


def snap(name):
    path = os.path.join(out_dir, name)
    r = mujoco.Renderer(world, 480, 640)
    r.update_scene(wdata)
    Image.fromarray(r.render()).save(path)
    r.close()
    print("wrote", path)


def report(tag):
    d = float(np.linalg.norm(0.5 * (wdata.xpos[lf] + wdata.xpos[rf]) - wdata.xpos[cube]) * 100)
    print(
        tag,
        "fingers_to_cube_cm",
        round(d, 1),
        "cube_z",
        round(float(wdata.xpos[cube][2]), 3),
        "finger_z",
        round(float(0.5 * (wdata.xpos[lf][2] + wdata.xpos[rf][2])), 3),
    )


def ik_to(xyz, iters=800):
    target = np.asarray(xyz, dtype=float)
    tip.set_target(SE3.from_translation(target))
    for _ in range(iters):
        vel = mink.solve_ik(configuration, tasks, dt, "daqp", damping=1e-3)
        configuration.integrate_inplace(vel, dt)
    apply_q()
    wdata.ctrl[0:7] = configuration.q[:7]
    wdata.ctrl[7] = 255
    err = float(np.linalg.norm(wdata.xpos[lf] - target) * 100)
    print("ik_err_cm", round(err, 1))


mujoco.mj_resetData(world, wdata)
apply_q()
report("home")
cube_pos = wdata.xpos[cube].copy()

# Hover 10 cm above the cube, then onto the cube (fingertip). Placement is
# mj_forward only — contacts do not run.
ik_to(cube_pos + np.array([0.0, 0.0, 0.10]))
report("hover")

ik_to(cube_pos + np.array([0.0, 0.0, 0.01]))
report("at_cube")
snap("pitfall-teleport.png")

hold = configuration.q[:7].copy()
for g in np.linspace(255, 0, 220):
    wdata.ctrl[0:7] = hold
    wdata.ctrl[7] = float(g)
    mujoco.mj_step(world, wdata)
report("closed")

lift = hold.copy()
lift[1] = hold[1] - 0.35
for _ in range(350):
    wdata.ctrl[0:7] = lift
    wdata.ctrl[7] = 0
    mujoco.mj_step(world, wdata)
report("lift")
if wdata.xpos[cube][2] > 0.08:
    print("GRASP_OK")
else:
    print("GRASP_FAIL")
print("OK")
