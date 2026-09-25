# -*- coding: utf-8 -*-
"""
Program E3: Coupled equations

    dx/dt =  y        (27a)
    dy/dt = -x        (27b)

Built on Program E2: same pattern (a function defining the right-hand
side, and x = x + h * f(...) stepping), extended to a *pair* of coupled
functions fx(x,y,t) and fy(x,y,t), one for each variable, updated
together each step.

Exact solution (simple harmonic oscillator, x(0)=0, y(0)=1):
    x(t) = sin(t)
    y(t) = cos(t)

@author: lukeh
"""

import matplotlib.pyplot as plt
import numpy as np


def fx(x, y, t):
    return y


def fy(x, y, t):
    return -x


def euler_coupled(x0, y0, h, t_max, t0=0.0, blowup=50):
    """
    Euler solver for the coupled system dx/dt = fx(x,y,t), dy/dt = fy(x,y,t).
    Both updates use the OLD x, y (standard explicit Euler - the two
    equations are stepped together, not one after the other).
    Stops early if |x| or |y| exceeds `blowup`, matching the guard used
    in Program E2.
    """
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]
    t, x, y = t0, x0, y0
    while abs(t - t_max) > h / 2:
        try:
            x_new = x + h * fx(x, y, t)
            y_new = y + h * fy(x, y, t)
        except OverflowError:
            break
        if not (np.isfinite(x_new) and np.isfinite(y_new)) or \
           abs(x_new) > blowup or abs(y_new) > blowup:
            break
        x, y = x_new, y_new
        t += h
        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
    return np.array(t_values), np.array(x_values), np.array(y_values)


# ---------------------------------------------------------------------
# Run for several step sizes, h = 0.05 down to h = 0.0005
# t_max covers at least 5 full cycles of the sine wave (period = 2*pi)
# ---------------------------------------------------------------------
h_values = [0.05, 0.01, 0.005, 0.001, 0.0005]
t_max = 5 * 2 * np.pi

# --- one panel per h, Euler x(t) vs exact x=sin(t) ---------------------
fig, axes = plt.subplots(len(h_values), 1, figsize=(8, 2.6 * len(h_values)),
                          sharex=True)

for ax, h in zip(axes, h_values):
    t_vals, x_vals, y_vals = euler_coupled(x0=0, y0=1, h=h, t_max=t_max)
    ax.plot(t_vals, x_vals, linewidth=1, label=f"Euler, h={h}")
    ax.plot(t_vals, np.sin(t_vals), "k--", linewidth=1, label="exact: x=sin(t)")
    ax.set_ylabel("x(t)")
    ax.set_title(f"h = {h}")
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_xlabel("t")
fig.suptitle("Program E3: Euler's method vs exact solution, dx/dt=y, dy/dt=-x", y=1.001)
plt.tight_layout()
plt.show()

# --- all h values overlaid on one graph, for direct comparison --------
plt.figure(figsize=(9, 5))
for h in h_values:
    t_vals, x_vals, y_vals = euler_coupled(x0=0, y0=1, h=h, t_max=t_max)
    plt.plot(t_vals, x_vals, linewidth=1, label=f"h={h}")

t_exact = np.linspace(0, t_max, 2000)
plt.plot(t_exact, np.sin(t_exact), "k--", linewidth=2, label="exact: x=sin(t)")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Euler's method amplitude growth vs step size h")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("All Program E3 figures generated.")