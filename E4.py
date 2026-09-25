# -*- coding: utf-8 -*-
"""
Program E4: Modified Euler method

Repeats the simple harmonic oscillator problem from Program E3

    dx/dt = v
    dv/dt = -x

but replaces plain (forward) Euler with the Modified Euler / improved
Euler method (a predictor-corrector, a.k.a. Heun's method):

    predictor:  xinit = x(i) + h*fx(x(i), v(i))
                vinit = v(i) + h*fv(x(i), v(i))

    corrector:  x(i+1) = x(i) + 0.5*h*(fx(x(i),v(i)) + fx(xinit,vinit))
                v(i+1) = v(i) + 0.5*h*(fv(x(i),v(i)) + fv(xinit,vinit))

As in E2/E3, fx and fv are defined as separate functions so the method
can be reused for a different problem just by swapping them out.

Exact solution (x0=0, v0=1):
    x(t) = sin(t)
    v(t) = cos(t)

@author: lukeh
"""

import matplotlib.pyplot as plt
import numpy as np


def fx(x, v, t):
    return v


def fv(x, v, t):
    return -x


def modified_euler(x0, v0, h, t_max, t0=0.0, blowup=50):
    """
    Modified Euler (predictor-corrector / Heun's method) for the coupled
    system dx/dt = fx(x,v,t), dv/dt = fv(x,v,t).
    """
    t_values = [t0]
    x_values = [x0]
    v_values = [v0]
    t, x, v = t0, x0, v0

    while abs(t - t_max) > h / 2:
        # predictor step (plain Euler)
        xinit = x + h * fx(x, v, t)
        vinit = v + h * fv(x, v, t)

        # corrector step (average of derivative at start and predicted end)
        x_new = x + 0.5 * h * (fx(x, v, t) + fx(xinit, vinit, t + h))
        v_new = v + 0.5 * h * (fv(x, v, t) + fv(xinit, vinit, t + h))

        if not (np.isfinite(x_new) and np.isfinite(v_new)) or \
           abs(x_new) > blowup or abs(v_new) > blowup:
            break

        x, v = x_new, v_new
        t += h

        t_values.append(t)
        x_values.append(x)
        v_values.append(v)

    return np.array(t_values), np.array(x_values), np.array(v_values)


# ---------------------------------------------------------------------
# Run for the same range of step sizes as Program E3
# t_max covers at least 5 full cycles of the sine wave (period = 2*pi)
# ---------------------------------------------------------------------
h_values = [0.05, 0.01, 0.005, 0.001, 0.0005]
t_max = 5 * 2 * np.pi

# --- one panel per h: Modified Euler x(t) and v(t) vs exact -----------
fig, axes = plt.subplots(len(h_values), 1, figsize=(8, 2.6 * len(h_values)),
                          sharex=True)

for ax, h in zip(axes, h_values):
    t_vals, x_vals, v_vals = modified_euler(x0=0, v0=1, h=h, t_max=t_max)
    ax.plot(t_vals, x_vals, linewidth=1, label=f"Mod. Euler x, h={h}")
    ax.plot(t_vals, v_vals, linewidth=1, label=f"Mod. Euler v, h={h}")
    ax.plot(t_vals, np.sin(t_vals), "k--", linewidth=1, label="exact x=sin(t)")
    ax.plot(t_vals, np.cos(t_vals), "grey", linestyle=":", linewidth=1, label="exact v=cos(t)")
    ax.set_ylabel("x, v")
    ax.set_title(f"h = {h}")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

axes[-1].set_xlabel("t")
fig.suptitle("Program E4: Modified Euler vs exact solution, dx/dt=v, dv/dt=-x", y=1.001)
plt.tight_layout()
plt.show()

# --- x(t) for all h overlaid, for direct comparison to E3's growth ----
plt.figure(figsize=(9, 5))
for h in h_values:
    t_vals, x_vals, v_vals = modified_euler(x0=0, v0=1, h=h, t_max=t_max)
    plt.plot(t_vals, x_vals, linewidth=1, label=f"h={h}")

t_exact = np.linspace(0, t_max, 2000)
plt.plot(t_exact, np.sin(t_exact), "k--", linewidth=2, label="exact: x=sin(t)")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Modified Euler: amplitude behaviour vs step size h")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("All Program E4 figures generated.")