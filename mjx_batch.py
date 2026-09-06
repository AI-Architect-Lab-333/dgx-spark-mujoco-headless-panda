#!/usr/bin/env python3
"""CPU one-env vs MJX batched Panda. Warp probe. Fail closed on numbers."""
import os
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

xml = os.path.expanduser(
    "~/inference/mujoco_menagerie/franka_emika_panda/mjx_scene.xml"
)
N_STEPS = 200
CPU_STEPS = 20000
BATCHES = (1, 8, 32, 128, 512, 1024, 2048, 4096)

print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
print("mujoco", mujoco.__version__)

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
print("nq", model.nq, "nv", model.nv, "nbody", model.nbody)

# --- CPU, one env ---
mujoco.mj_resetData(model, data)
for _ in range(20):
    mujoco.mj_step(model, data)
t0 = time.perf_counter()
for _ in range(CPU_STEPS):
    mujoco.mj_step(model, data)
cpu_s = time.perf_counter() - t0
cpu_sps = CPU_STEPS / cpu_s
print("cpu_one", "steps", CPU_STEPS, "sec", round(cpu_s, 4), "steps_per_sec", round(cpu_sps, 1))

# --- MJX ---
mx = mjx.put_model(model)
dx0 = mjx.put_data(model, data)


def run_batch(n):
    rng = jax.random.split(jax.random.PRNGKey(0), n)
    # Tiny qpos jitter so XLA cannot collapse identical worlds.
    batch = jax.vmap(
        lambda k: dx0.replace(qpos=dx0.qpos + 1e-4 * jax.random.normal(k, dx0.qpos.shape))
    )(rng)
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    t_jit = time.perf_counter()
    batch = step(mx, batch)
    batch.time.block_until_ready()
    jit_s = time.perf_counter() - t_jit

    t1 = time.perf_counter()
    for _ in range(N_STEPS):
        batch = step(mx, batch)
    batch.time.block_until_ready()
    elapsed = time.perf_counter() - t1
    env_sps = n * N_STEPS / elapsed
    print(
        "mjx_batch",
        n,
        "jit_sec",
        round(jit_s, 3),
        "sec",
        round(elapsed, 4),
        "env_steps_per_sec",
        round(env_sps, 1),
        "per_env_steps_per_sec",
        round(N_STEPS / elapsed, 1),
        "vs_cpu_one",
        round(env_sps / cpu_sps, 2),
    )
    return env_sps


print("mjx_batches", BATCHES)
best_n = None
best_sps = 0.0
for n in BATCHES:
    try:
        sps = run_batch(n)
        if sps > best_sps:
            best_sps = sps
            best_n = n
    except Exception as e:
        print("mjx_batch_fail", n, type(e).__name__, str(e).split("\n")[0][:200])
        traceback.print_exc()
        break

print("mjx_best_n", best_n, "mjx_best_env_steps_per_sec", round(best_sps, 1))
if best_n is None:
    print("MJX_BATCH_FAIL")
elif best_sps > cpu_sps:
    print("MJX_BEATS_CPU")
else:
    print("MJX_STILL_SLOWER_THAN_CPU")

# --- Warp ---
print("warp_probe")
try:
    import warp

    print("warp", getattr(warp, "__version__", "?"))
except Exception as e:
    print("warp_import_fail", type(e).__name__, e)

try:
    from mujoco.mjx import warp as mjx_warp

    print("mjx_warp_ok", mjx_warp)
except Exception as e:
    print("mjx_warp_fail", type(e).__name__, e)

try:
    import mujoco_warp as mjw

    print("mujoco_warp", getattr(mjw, "__version__", "?"))
except Exception as e:
    print("mujoco_warp_fail", type(e).__name__, e)

print("OK")
