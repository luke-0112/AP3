# -*- coding: utf-8 -*-
"""
Program E2: A non-linear example

    dx/dt = t - x^2,  with x = x0 at t = 0        (15)

Built on Program E1: replaces the direct calculation with a function
f(x, t) = t - x^2, and uses x = x + h * f(x, t) to step forward, so f
can be swapped out later (e.g. for Program E3's coupled system).

@author: lukeh
"""

import matplotlib.pyplot as plt
import numpy as np


def f(x, t):
    return t - x**2


def euler(x0, h, t_max, t0=0.0, blowup=50):
    """
    Basic Euler solver for dx/dt = f(x,t).
    Stops early (instead of crashing) if |x| exceeds `blowup`, so that
    diverging/chaotic trajectories can still be plotted up to the point
    they blow up.
    """
    t_values = [t0]
    x_values = [x0]
    t, x = t0, x0
    while abs(t - t_max) > h / 2:
        try:
            x = x + h * f(x, t)
        except OverflowError:
            break
        if not np.isfinite(x) or abs(x) > blowup:
            break
        t += h
        t_values.append(t)
        x_values.append(x)
    return np.array(t_values), np.array(x_values)


# ---------------------------------------------------------------------
# 1. Baseline run: h = 0.05, x0 = 1, t_max = 9
# ---------------------------------------------------------------------
h = 0.05
t_max = 9
t_vals, x_vals = euler(x0=1, h=h, t_max=t_max)

plt.figure(figsize=(7, 5))
plt.plot(t_vals, x_vals)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title(f"Program E2 baseline: x0=1, h={h}, t_max={t_max}")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# 2. Experiment with other starting values, same h and t_max
#    Note: x0 > -0.75 converges onto the same curve; x0 = -0.75 diverges
# ---------------------------------------------------------------------
initial_values = [3, 1, 0, -0.7, -0.75]

plt.figure(figsize=(7, 5))
for x0 in initial_values:
    t_vals, x_vals = euler(x0=x0, h=h, t_max=t_max)
    plt.plot(t_vals, x_vals, label=f"x0 = {x0}")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title(f"Varying x0, h={h}, t_max={t_max}")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# 3. Longer run times: t_max in {50, 100, 500, 1000}
#    At medium t_max, x should converge onto x = sqrt(t).
#    At higher t_max, it diverges and eventually becomes chaotic.
# ---------------------------------------------------------------------
t_max_values = [50, 100, 500, 1000]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for ax, tmax in zip(axes, t_max_values):
    for x0 in initial_values:
        t_vals, x_vals = euler(x0=x0, h=h, t_max=tmax)
        ax.plot(t_vals, x_vals, linewidth=1, label=f"x0={x0}")
    t_ref = np.linspace(0.01, tmax, 500)
    ax.plot(t_ref, np.sqrt(t_ref), "k--", linewidth=1.5, label="sqrt(t)")
    ax.set_title(f"t_max = {tmax}")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.legend(fontsize=7)

fig.suptitle(f"Longer run times, h={h}")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# 4. Rerun with a smaller step size, h = 0.01, same t_max sweep
#    Smaller h should delay/remove the divergence seen above.
# ---------------------------------------------------------------------
h_small = 0.01

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for ax, tmax in zip(axes, t_max_values):
    for x0 in initial_values:
        t_vals, x_vals = euler(x0=x0, h=h_small, t_max=tmax)
        ax.plot(t_vals, x_vals, linewidth=1, label=f"x0={x0}")
    t_ref = np.linspace(0.01, tmax, 500)
    ax.plot(t_ref, np.sqrt(t_ref), "k--", linewidth=1.5, label="sqrt(t)")
    ax.set_title(f"t_max = {tmax}")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.legend(fontsize=7)

fig.suptitle(f"Longer run times, smaller step size h={h_small}")
plt.tight_layout()
plt.show()

print("All Program E2 figures generated.")