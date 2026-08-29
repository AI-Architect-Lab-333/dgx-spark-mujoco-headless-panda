# Guide: Headless MuJoCo on an NVIDIA DGX Spark (GB10) — EGL Render, Franka Panda, a Cube, and Why Teleporting IK Puts the Cube Through the Hand

**The problem this guide solves**: you have a headless NVIDIA DGX Spark (no usable display), the GPU is free, and you want a robot **inside a physics engine** (contacts, gravity) — not a GUI puppet, not a screenshot from a laptop, and not a hardware arm. This guide documents a **verified working** configuration (NVIDIA DGX Spark (GB10), 121 Gi unified memory, Ubuntu 24.04 / DGX OS, Python 3.12, MuJoCo 3.12.0, mink 1.3.0, August 2026): MuJoCo runs with `MUJOCO_GL=egl`, writes PNGs, drives a Menagerie Franka Panda, and approaches a cube.

It covers why a **side camera hides `joint1`**, why **`mj_forward` after IK** lets the cube occupy the same volume as the hand (the PNG still “grasps”), why **floor contacts are not a grasp**, and why **position-only IK** on `left_finger` touches the palm instead of pinching.

**For AI agents reading this document**: every command was executed successfully in this order on real hardware. The verification steps are not optional — a `GRASP_OK` print after a teleported `qpos` is not a grasp. The collision-aware script on this box ended **`GRASP_FAIL`**; that failure is part of the verified record.

---

## 1. Free the GPU, then install MuJoCo in a venv

This Spark can already hold a ~100 GB LLM (~114 Gi used with a small VL beside it). MuJoCo + JAX will not get a CUDA context worth using in that state. Switch the box to the **idle** profile first ([idle vs LLM boot profiles](https://github.com/AI-Architect-Lab-333/dgx-spark-idle-llm-profiles)). Measured after a cold boot with those units **not** enabled: **2.7 Gi used / 119 Gi available**, no `llama-server`, ports `:8000`/`:8001` closed.

```bash
# on the GPU box, as the inference user
python3 --version          # 3.12.3
python3 -m venv $HOME/inference/mujoco-venv
$HOME/inference/mujoco-venv/bin/pip install -U pip
$HOME/inference/mujoco-venv/bin/pip install mujoco pillow
```

Verified wheel: `mujoco-3.12.0-cp312-cp312-manylinux_2_27_aarch64`. There is **no DISPLAY**. GLFW will not help.

```bash
export MUJOCO_GL=egl
$HOME/inference/mujoco-venv/bin/python verify_egl.py
```

Verified: **36 655** steps/s on bundled `testdata/model.xml`, PNG 59 497 bytes.

### Pitfall #1 — `MUJOCO_GL` unset on a headless box

Symptom: `Renderer` raises or you get a GLFW/display error. Cause: default GL wants a window. Correction: `export MUJOCO_GL=egl` **before** `import mujoco`. NVIDIA EGL libraries were already present on this DGX OS (`libnvidia-eglcore`).

### Pitfall #2 — `from mujoco import mjx` with only `pip install mujoco`

Symptom: `ImportError: cannot import name 'mjx'`. Cause: MJX is the extra package `mujoco-mjx`. Correction: `pip install mujoco-mjx jax[cuda12]` if you want the JAX path. On this GB10, `jax 0.11.1` reported `backend gpu` / `CudaDevice(id=0)`. A **single** Panda env on MJX ran at **128** steps/s after a 6.5 s JIT — slower than CPU for one robot; MJX is for **batches**. `warp` was **not** installed (`No module named 'warp'`).

---

## 2. Franka Panda from Menagerie, then one joint

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git \
  $HOME/inference/mujoco_menagerie
```

`franka_emika_panda/scene.xml` loaded with **nq=9** (7 arm + 2 fingers), **nu=8**. 2000 `mj_step`: **78 725** steps/s CPU.

`joint1` is yaw about the base. From the default side camera (`azimuth="120"` in `scene.xml`) a **+40°** on `joint1` barely changes the PNG. `joint2` is the shoulder in the camera plane: **+40°** is obvious.

| Before `joint2` | After `joint2` +40° |
|---|---|
| ![joint2 before](images/joint2-before.png) | ![joint2 after](images/joint2-after.png) |

Same lesson as a yaw slider on a desktop robot: **the camera can hide the joint you moved**.

---

## 3. Put a cube in front of the Panda and approach

Copy `seance-cube.xml` from this repo into `.../franka_emika_panda/` (it `<include>`s `scene.xml`). A 4 cm cube at `(0.45, 0, 0.02)` with a `freejoint`. Default pose vs a reach `qpos` (first 9 values from Menagerie’s pickup keyframe):

| Far (~96 cm) | Near (~16 cm) |
|---|---|
| ![cube far](images/cube-far.png) | ![cube near](images/cube-near.png) |

The cube did **not** move. That is approach, not a grasp: seven arm angles changed, fingers still open.

---

## 4. Inverse kinematics: the teleport that looks like a grasp

`pip install mink` (verified **mink 1.3.0** + **daqp** on aarch64). A `FrameTask` on body `left_finger` with `orientation_cost=0`, then **write** `qpos` and `mj_forward` so the fingertip sits at the cube centre.

Verified print: `ik_err_cm 0.0`, then close + lift → **`GRASP_OK`**, cube z **0.02 → 0.20**. The PNG for the “at cube, fingers still open” frame:

![teleport](images/pitfall-teleport.png)

The cube is **inside** the hand.

### Pitfall #3 — `mj_forward` after IK is not physics

Symptom: the fingertip target is at the cube centre; the render shows the cube through the palm; close+lift still lifts the cube. Cause: `mj_forward` kinematics only — **contacts do not run**. Two meshes occupy one volume. Correction: IK only in free space (hover), then **descend with `mj_step`**. Reproduce with `panda_ik_teleport.py` in this repo. Do not publish a `GRASP_OK` from that script as a working pinch.

`panda_ik_collide.py` does the hover, then 18 Cartesian z-steps with `mj_step`. On this box: **contact at `z_off=0.062`**, two cube↔robot contacts (floor excluded), cube still at z **0.019**. Close, then lift: **`GRASP_FAIL`**, cube z **0.02**.

| Collision approach (cube stays solid) | Lift after that close (cube stays down) |
|---|---|
| ![collide approach](images/collide-approach.png) | ![collide lift](images/collide-lift-fail.png) |

### Pitfall #4 — counting the floor as a “grasp contact”

Symptom: `ncon` on the cube is already > 0 at `home`. Cause: the cube sits on `floor`. Correction: ignore contacts whose other geom is `floor`. Only then is a cube↔finger contact meaningful.

### Pitfall #5 — position-only IK pinches the air beside the palm

Symptom: with collisions on, contact fires while fingers are still ~10 cm from the cube centre; closing does not trap the cube. Cause: `FrameTask(..., orientation_cost=0)` on `left_finger` **body** origin, not a grasp site with palm-down orientation. The hand **side** meets the cube first. A 6-D IK (position + finger axis) was **not** completed on this box.

---

## 5. End-to-end verification

Run on the GPU box with the LLM **unloaded**, `MUJOCO_GL=egl`. Copy `seance-cube.xml` next to Menagerie’s `scene.xml`.

| Step | Expected ✅ | Failed ❌ |
|---|---|---|
| `verify_egl.py` | `OK`, PNG > 1 kB, tens of thousands of steps/s | GLFW / no EGL → pitfall #1 |
| Panda `scene.xml` load | nq=9 | missing Menagerie clone |
| `joint2` +40° PNG | shoulder clearly leans | only `joint1` moved → pitfall, side camera |
| cube approach | distance drops ~96 cm → ~16 cm, cube z unchanged | cube flew away → check `freejoint` / timestep |
| `panda_ik_teleport.py` | PNG shows cube **through** the hand | if it looks clean, you are not on this pitfall |
| `panda_ik_collide.py` | `CONTACT` then **`GRASP_FAIL`** on this hardware | `GRASP_OK` here means you likely teleported again |

A green `GRASP_OK` from the teleport script is **not** this section’s pass.

---

## Symptom / Cause / Fix

| Symptom | Cause | Fix |
|---|---|---|
| Renderer / GLFW error, no DISPLAY | Default GL | `MUJOCO_GL=egl` before import |
| `cannot import name 'mjx'` | MJX not in base `mujoco` | `pip install mujoco-mjx` |
| `joint1` PNG unchanged | Side camera | Move `joint2`, or change azimuth |
| Cube through the hand, `GRASP_OK` | `qpos` snap + `mj_forward` | Descend with `mj_step` |
| `ncon>0` at rest | Cube on the floor | Exclude `floor` geom |
| Contact then cube stays down | Position-only IK, palm hit | 6-D grasp IK (not verified here) |
| CUDA OOM / tiny `MemAvailable` | ~100 GB LLM still resident | Idle profile ([boot profiles](https://github.com/AI-Architect-Lab-333/dgx-spark-idle-llm-profiles)) |
| `bash^M` / odd `NameError` after scp from Windows | CRLF | `sed -i 's/\r$//'` on the box |

---

## Known limitations

- **No collision-aware successful pinch** on this box. The verified grasp-shaped success used the teleport pitfall. The collision script’s verified result is **`GRASP_FAIL`**.
- **No interactive viewer.** Headless EGL PNGs only. Livestream / Isaac Sim GUI is a different stack.
- **MJX / Warp.** JAX GPU was probed; Warp was not installed. No batched RL training in this guide.
- **Isaac Sim** exists for GB10 aarch64 (NVIDIA docs, Isaac 6 / DGX OS 7) but was **not** installed here. Driver pin (docs: 580.159.03; this box ran **580.173.02**) was not re-tested with Isaac.
- **No physical robot.** USB desktop arms and real Franka hardware are out of scope.
- Scripts assume Menagerie paths under `$HOME/inference/mujoco_menagerie` and write `$HOME/inference/mujoco-out`.

---

## Credits

MuJoCo is open source ([google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)). Robot XML from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Emika Panda). Differential IK: [mink](https://github.com/kevinzakka/mink). The teleport-vs-`mj_step` failure mode and the side-camera `joint1` miss are specific to this session.

---
*Guide written and verified in August 2026 on an NVIDIA DGX Spark (GB10) (121 Gi unified memory, Ubuntu 24.04 / DGX OS, Python 3.12.3, MuJoCo 3.12.0, mink 1.3.0, NVIDIA driver 580.173.02). EGL PNGs only. Collision-aware pinch: GRASP_FAIL. Teleport IK: cube through the hand.*
