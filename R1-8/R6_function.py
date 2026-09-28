import numpy as np

def driven_pendulum(t, y, b, omega0, omegad, A):
    x, v = y
    dxdt = v
    dvdt = -b * v - (omega0**2) * x - A * np.sin(omegad * t)
    dydt = np.array([dxdt, dvdt])
    return dydt