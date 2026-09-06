#!/usr/bin/env python3
"""MJWarp batch probe on the same Panda MJX scene. Not a training run."""
import os
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import warp as wp
import mujoco_warp as mjw

xml = os.path.expanduser(
    "~/inference/mujoco_menagerie/franka_emika_panda/mjx_scene.xml"
)
N_STEPS = 200
BATCHES = (1, 128, 1024, 2048, 4096)

print("warp", wp.__version__)
wp.init()
print("warp_init_ok")
print("mujoco", mujoco.__version__, "mujoco_warp", getattr(mjw, "__version__", "?"))

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
m = mjw.put_model(model)


def run_batch(n):
    d = mjw.put_data(model, data, nworld=n)
    t_jit = time.perf_counter()
    mjw.step(m, d)
    wp.synchronize()
    jit_s = time.perf_counter() - t_jit
    t1 = time.perf_counter()
    for _ in range(N_STEPS):
        mjw.step(m, d)
    wp.synchronize()
    elapsed = time.perf_counter() - t1
    env_sps = n * N_STEPS / elapsed
    print(
        "mjwarp_batch",
        n,
        "jit_sec",
        round(jit_s, 3),
        "sec",
        round(elapsed, 4),
        "env_steps_per_sec",
        round(env_sps, 1),
        "per_env_steps_per_sec",
        round(N_STEPS / elapsed, 1),
    )
    return env_sps


best_n = None
best_sps = 0.0
for n in BATCHES:
    try:
        sps = run_batch(n)
        if sps > best_sps:
            best_sps = sps
            best_n = n
    except Exception as e:
        print("mjwarp_fail", n, type(e).__name__, str(e).split("\n")[0][:240])
        traceback.print_exc()
        break

print("mjwarp_best_n", best_n, "mjwarp_best_env_steps_per_sec", round(best_sps, 1))
print("OK")
