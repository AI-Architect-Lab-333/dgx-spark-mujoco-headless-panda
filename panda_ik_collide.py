#!/usr/bin/env python3
"""IK to a hover pose, then descend with mj_step so the cube cannot pass through
the hand. Close fingers, lift. Prints GRASP_OK only if the cube leaves the floor.
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
rf = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
cube_body = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_geom = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_GEOM, "cube")
floor_geom = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_GEOM, "floor")

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


def cube_robot_contacts():
    n = 0
    for i in range(wdata.ncon):
        pair = {int(wdata.contact[i].geom1), int(wdata.contact[i].geom2)}
        if cube_geom in pair and floor_geom not in pair:
            n += 1
    return n


def report(tag):
    mid = 0.5 * (wdata.xpos[lf] + wdata.xpos[rf])
    d = float(np.linalg.norm(mid - wdata.xpos[cube_body]) * 100)
    print(
        tag,
        "fingers_to_cube_cm",
        round(d, 1),
        "cube_z",
        round(float(wdata.xpos[cube_body][2]), 3),
        "ncon_cube_robot",
        cube_robot_contacts(),
    )


def ik_set(xyz, iters=500):
    tip.set_target(SE3.from_translation(np.asarray(xyz, dtype=float)))
    for _ in range(iters):
        vel = mink.solve_ik(configuration, tasks, 0.01, "daqp", damping=1e-3)
        configuration.integrate_inplace(vel, 0.01)
    wdata.ctrl[0:7] = configuration.q[:7]
    wdata.ctrl[7] = 255


def hold_step(n, grip):
    q7 = configuration.q[:7].copy()
    for _ in range(n):
        wdata.ctrl[0:7] = q7
        wdata.ctrl[7] = grip
        mujoco.mj_step(world, wdata)


mujoco.mj_resetData(world, wdata)
configuration.update_from_keyframe("home")
wdata.qpos[0:9] = configuration.q[:9]
wdata.ctrl[0:7] = configuration.q[:7]
wdata.ctrl[7] = 255
mujoco.mj_forward(world, wdata)
cube0 = wdata.xpos[cube_body].copy()
report("home")

ik_set(cube0 + np.array([0.0, 0.0, 0.12]), iters=800)
hold_step(40, 255)
configuration.update(wdata.qpos[: robot.nq])
report("hover")
snap("collide-hover.png")

contacted = False
for z_off in np.linspace(0.12, 0.03, 18):
    ik_set(cube0 + np.array([0.0, 0.0, float(z_off)]), iters=120)
    hold_step(25, 255)
    configuration.update(wdata.qpos[: robot.nq])
    if cube_robot_contacts() > 0:
        contacted = True
        print("contact at z_off", round(float(z_off), 3))
        break
report("approach")
print("CONTACT" if contacted else "NO_CONTACT")
snap("collide-approach.png")

hold = configuration.q[:7].copy()
for g in np.linspace(255, 0, 240):
    wdata.ctrl[0:7] = hold
    wdata.ctrl[7] = float(g)
    mujoco.mj_step(world, wdata)
report("closed")
snap("collide-closed.png")

lift = hold.copy()
lift[1] = hold[1] - 0.35
for _ in range(400):
    wdata.ctrl[0:7] = lift
    wdata.ctrl[7] = 0
    mujoco.mj_step(world, wdata)
report("lift")
print("GRASP_OK" if wdata.xpos[cube_body][2] > 0.08 else "GRASP_FAIL")
snap("collide-lift.png")
print("OK")
