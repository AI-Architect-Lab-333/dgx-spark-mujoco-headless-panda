#!/usr/bin/env python3
"""6-D grasp IK (position + palm-down orientation), then a collision-aware pinch.

Hover in free space, descend with mj_step (never snap qpos into the cube),
close, lift. Fail closed if the cube stays on the floor.

The previous collide script used a position-only FrameTask on left_finger, so
the palm hit the cube from the side. This one drives the hand frame: TCP at
the fingertip pads, Z through the fingers, Y along the closing axis.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mink
import mujoco
import numpy as np
from PIL import Image
from mink import SE3, SO3

robot_xml = os.path.expanduser(
    "~/inference/mujoco_menagerie/franka_emika_panda/scene.xml"
)
_panda_dir = os.path.expanduser("~/inference/mujoco_menagerie/franka_emika_panda")
world_xml = os.path.join(_panda_dir, "panda-cube.xml")
if not os.path.isfile(world_xml):
    world_xml = os.path.join(_panda_dir, "seance-cube.xml")
out_dir = os.path.expanduser("~/inference/mujoco-out")
os.makedirs(out_dir, exist_ok=True)

# Pad centre in the hand frame: finger body at z=0.0584, pad at z=0.0445.
TCP_IN_HAND = np.array([0.0, 0.0, 0.1029])

robot = mujoco.MjModel.from_xml_path(robot_xml)
world = mujoco.MjModel.from_xml_path(world_xml)
# Stiffer friction constraints so a pinch can carry the cube on lift.
world.opt.impratio = 10
wdata = mujoco.MjData(world)
fj1 = int(world.jnt_qposadr[mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")])
fj2 = int(world.jnt_qposadr[mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")])
hand = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "hand")
lf = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
cube_body = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_geom = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_GEOM, "cube")
floor_geom = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_GEOM, "floor")

configuration = mink.Configuration(robot)
configuration.update_from_keyframe("home")
ee = mink.FrameTask(
    "hand",
    "body",
    position_cost=1.0,
    orientation_cost=1.0,
    lm_damping=1e-2,
)
posture = mink.PostureTask(robot, cost=1e-3)
posture.set_target_from_configuration(configuration)
tasks = [ee, posture]
limits = [mink.ConfigurationLimit(robot)]
ik_dt = 0.01


def grasp_so3():
    """Hand axes in world: X = ortho, Y = closing, Z = approach (down)."""
    approaching = np.array([0.0, 0.0, -1.0])
    closing = np.array([0.0, 1.0, 0.0])
    ortho = np.cross(closing, approaching)
    R = np.column_stack((ortho, closing, approaching))
    return SO3.from_matrix(R)


GRASP_R = grasp_so3()
GRASP_RM = GRASP_R.as_matrix()


PAD_IN_FINGER = np.array([0.0, 0.0055, 0.0445])


def tcp_world():
    R = wdata.xmat[hand].reshape(3, 3)
    return wdata.xpos[hand] + R @ TCP_IN_HAND


def pad_center(finger_id):
    R = wdata.xmat[finger_id].reshape(3, 3)
    return wdata.xpos[finger_id] + R @ PAD_IN_FINGER


def hand_xyz_for_tcp(tcp_xyz):
    return np.asarray(tcp_xyz, dtype=float) - GRASP_RM @ TCP_IN_HAND


def ori_err_deg():
    R_cur = wdata.xmat[hand].reshape(3, 3)
    cosang = 0.5 * (np.trace(GRASP_RM @ R_cur.T) - 1.0)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def cube_contacts_by_body():
    left = right = other = 0
    for i in range(wdata.ncon):
        c = wdata.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if cube_geom not in pair:
            continue
        gid = next(iter(pair - {cube_geom}))
        if gid == floor_geom:
            continue
        bname = mujoco.mj_id2name(
            world, mujoco.mjtObj.mjOBJ_BODY, int(world.geom_bodyid[gid])
        )
        if bname == "left_finger":
            left += 1
        elif bname == "right_finger":
            right += 1
        else:
            other += 1
    return left, right, other


def cube_contact_count():
    left, right, other = cube_contacts_by_body()
    return left + right + other


def finger_mid():
    return 0.5 * (wdata.xpos[lf] + wdata.xpos[rf])


def report(tag):
    tcp = tcp_world()
    cube = wdata.xpos[cube_body]
    print(
        tag,
        "tcp_to_cube_cm",
        round(float(np.linalg.norm(tcp - cube) * 100), 1),
        "cube_xy",
        np.round(cube[:2], 4),
        "cube_z",
        round(float(cube[2]), 3),
        "tcp_z",
        round(float(tcp[2]), 3),
        "ori_err_deg",
        round(ori_err_deg(), 1),
        "finger_q",
        np.round(wdata.qpos[[fj1, fj2]], 4),
        "ncon_cube",
        cube_contact_count(),
    )
    lp, rp = pad_center(lf), pad_center(rf)
    print(
        " ",
        "hand",
        np.round(wdata.xpos[hand], 4),
        "tcp",
        np.round(tcp, 4),
        "pad_l",
        np.round(lp, 4),
        "pad_r",
        np.round(rp, 4),
        "pad_mid",
        np.round(0.5 * (lp + rp), 4),
    )


def dump_cube_contacts(tag):
    print(tag, "contacts")
    for i in range(wdata.ncon):
        c = wdata.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if cube_geom not in pair:
            continue
        other = next(iter(pair - {cube_geom}))
        if other == floor_geom:
            continue
        ob = int(world.geom_bodyid[other])
        bname = mujoco.mj_id2name(world, mujoco.mjtObj.mjOBJ_BODY, ob) or "?"
        gname = mujoco.mj_id2name(world, mujoco.mjtObj.mjOBJ_GEOM, other) or f"g{other}"
        force = np.zeros(6)
        mujoco.mj_contactForce(world, wdata, i, force)
        print(
            " ",
            bname,
            gname,
            "pos",
            np.round(c.pos, 4),
            "dist",
            round(float(c.dist), 5),
            "fn",
            round(float(force[0]), 3),
        )


def snap(name, azimuth=120, elevation=-20):
    path = os.path.join(out_dir, name)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(world, cam)
    cam.lookat[:] = wdata.xpos[cube_body]
    cam.distance = 0.55
    cam.azimuth = azimuth
    cam.elevation = elevation
    r = mujoco.Renderer(world, 480, 640)
    r.update_scene(wdata, camera=cam)
    Image.fromarray(r.render()).save(path)
    r.close()
    print("wrote", path)


def ik_to_tcp(tcp_xyz, iters=600):
    target = SE3.from_rotation_and_translation(GRASP_R, hand_xyz_for_tcp(tcp_xyz))
    ee.set_target(target)
    for _ in range(iters):
        vel = mink.solve_ik(
            configuration, tasks, ik_dt, "daqp", damping=1e-3, limits=limits
        )
        configuration.integrate_inplace(vel, ik_dt)


def apply_arm_qpos_free_space():
    """Snap the arm only. Cube freejoint is left untouched. Hover use only."""
    wdata.qpos[0:9] = configuration.q[:9]
    wdata.ctrl[0:7] = configuration.q[:7]
    wdata.ctrl[7] = 255
    mujoco.mj_forward(world, wdata)


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
print("cube", np.round(cube0, 4))
print("grasp_R")
print(np.round(GRASP_RM, 3))
report("home")

# Reach 1 cm further in +X: at the first run the pads sat on the -X edge
# of the cube (x≈0.442 vs 0.45) and the pinch levered the cube out.
grasp_xy = cube0 + np.array([0.01, 0.0, 0.0])

# 1. Hover: 6-D IK in free space, then a short physics settle.
hover_tcp = grasp_xy + np.array([0.0, 0.0, 0.12])
ik_to_tcp(hover_tcp, iters=900)
apply_arm_qpos_free_space()
hold_step(40, 255)
configuration.update(wdata.qpos[: robot.nq])
report("hover")
snap("ik6d-hover.png")

# 2. Descend in Cartesian TCP steps. Physics every step: no cube-through-hand.
contacted = False
contact_z = None
for z_off in np.linspace(0.12, 0.0, 25):
    ik_to_tcp(grasp_xy + np.array([0.0, 0.0, float(z_off)]), iters=250)
    hold_step(30, 255)
    configuration.update(wdata.qpos[: robot.nq])
    if cube_contact_count() > 0:
        contacted = True
        contact_z = float(z_off)
        print("contact at z_off", round(contact_z, 3))
        break

report("approach")
print("CONTACT" if contacted else "NO_CONTACT")
snap("ik6d-approach.png")

# 3. Close until both finger pads see the cube, then hold that width.
# Commanding 0 after that lets the fingers meet and pops the cube out.
hold = configuration.q[:7].copy()
grip_final = 0.0
pinched = False
for g in np.linspace(255, 0, 240):
    wdata.ctrl[0:7] = hold
    wdata.ctrl[7] = float(g)
    mujoco.mj_step(world, wdata)
    grip_final = float(g)
    left_c, right_c, _ = cube_contacts_by_body()
    drift = float(np.linalg.norm(wdata.xpos[cube_body][:2] - cube0[:2]))
    if drift > 0.015:
        print("cube pushed at grip", round(grip_final, 1), "drift_cm", round(drift * 100, 1))
        break
    if left_c > 0 and right_c > 0:
        pinched = True
        squeeze = max(0.0, grip_final - 15.0)
        for _ in range(60):
            wdata.ctrl[0:7] = hold
            wdata.ctrl[7] = squeeze
            mujoco.mj_step(world, wdata)
        grip_final = squeeze
        print("pinch at grip", round(grip_final, 1), "left", left_c, "right", right_c)
        break
print("PINCH" if pinched else "NO_PINCH")
hold_step(200, grip_final)
report("closed")
dump_cube_contacts("closed")
snap("ik6d-closed.png")
snap("ik6d-closed-side.png", azimuth=90, elevation=-15)

# 4. Slow +Z lift via 6-D IK + mj_step (still no qpos snap).
start_tcp = tcp_world().copy()
for z_add in np.linspace(0.0, 0.18, 19)[1:]:
    ik_to_tcp(start_tcp + np.array([0.0, 0.0, float(z_add)]), iters=80)
    hold_step(40, grip_final)
    configuration.update(wdata.qpos[: robot.nq])
report("lift")
dump_cube_contacts("lift")
held = (
    wdata.xpos[cube_body][2] > 0.08
    and cube_contact_count() > 0
    and float(np.linalg.norm(tcp_world() - wdata.xpos[cube_body])) < 0.05
)
if held:
    print("GRASP_OK")
    snap("ik6d-lift-ok.png")
    snap("ik6d-lift-ok-side.png", azimuth=90, elevation=-15)
else:
    print("GRASP_FAIL")
    snap("ik6d-lift-fail.png")
    snap("ik6d-lift-fail-side.png", azimuth=90, elevation=-15)
print("OK")
