import numpy as np

def driven_pendulum(t, y, b, omega_0, omega_d, A):
    x, v = y
    dxdt = v
    dvdt = -b * v - (omega_0**2) * x - A * np.sin(omega_d * t)
    dydt = np.array([dxdt, dvdt])
    return dydt