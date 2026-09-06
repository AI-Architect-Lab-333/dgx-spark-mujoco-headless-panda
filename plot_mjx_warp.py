#!/usr/bin/env python3
"""Render the MJX vs Warp vs CPU throughput figure from measured GB10 prints.

Numbers are copied from the 2026-09-06 Spark idle run (mjx_scene.xml, nq=9).
This script does not talk to the GPU.
"""
import os

import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(__file__), "images", "mjx-warp-throughput.png")

cpu = 216889.2
mjx_n = [1, 8, 32, 128, 512, 1024, 2048, 4096]
mjx_sps = [127.8, 1008.0, 4043.9, 15538.1, 61331.9, 120475.1, 229958.8, 414478.7]
warp_n = [1, 128, 1024, 2048, 4096]
warp_sps = [405.8, 51652.8, 405137.9, 791235.6, 1477178.9]

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.axhline(cpu, color="#666666", linestyle="--", linewidth=1.2, label="CPU, 1 Panda (216 889 steps/s)")
ax.plot(mjx_n, mjx_sps, "o-", color="#1f4e79", linewidth=1.8, markersize=6, label="MJX (JAX vmap)")
ax.plot(warp_n, warp_sps, "s-", color="#c45c26", linewidth=1.8, markersize=6, label="Warp (nworld)")
ax.axvline(2048, color="#1f4e79", linestyle=":", linewidth=0.9, alpha=0.7)
ax.axvline(1024, color="#c45c26", linestyle=":", linewidth=0.9, alpha=0.7)
ax.annotate(
    "MJX > CPU\n(N=2048)",
    xy=(2048, 229958.8),
    xytext=(2200, 4.5e4),
    fontsize=8,
    color="#1f4e79",
    arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=0.8),
)
ax.annotate(
    "Warp > CPU\n(N=1024)",
    xy=(1024, 405137.9),
    xytext=(140, 9e5),
    fontsize=8,
    color="#c45c26",
    arrowprops=dict(arrowstyle="->", color="#c45c26", lw=0.8),
)
ax.set_xticks(mjx_n)
ax.set_xticklabels([str(n) for n in mjx_n])
ax.set_xlabel("Parallel Panda environments (N)")
ax.set_ylabel("Env-steps / s  (N × steps / wall)")
ax.set_title("GB10 idle, mjx_scene.xml — GPU beats one CPU Panda only in batch")
ax.set_xlim(0.7, 5500)
ax.set_ylim(80, 3.2e6)
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
