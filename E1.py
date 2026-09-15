# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:03:32 2026

@author: lukeh
"""

import matplotlib.pyplot as plt

#variables
t_step = 0.1
t_max = 10
tau = 2.0

#initial conditions
t0 = 0
N0 = 10

#arrays for N and t
t_values = [t0]
N_values = [N0]

#start loop
t = t0
N = N0
while abs(t - t_max) > t_step / 2:
    N = N - N / tau * t_step
    t = t + t_step
    t_values.append(t)
    N_values.append(N)

#plot graph
plt.plot(t_values, N_values, 'o-', markersize=3)
plt.xlabel('Time, t')
plt.ylabel('N(t)')
plt.title("Radioactive Decay by Euler's Method")
plt.grid(True)
plt.show()