import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from R7_function import driven_pendulum

def main():

    A = 1.0
    b = 0.1
    omega_0 = 1.0
    
    x0 = 0.0
    v0 = 0.0
    y0 = (x0, v0)
    
    tf = 100.0
    n = 500    
    t_steady = np.linspace(0.8 * tf, tf, n)
    
    driving_freq = np.linspace(0.2 * omega_0, 2.0 * omega_0, 100)
    
    amplitudes = []
    
    for omega_d in driving_freq:
        
        lfun = lambda t, y: driven_pendulum(t, y, b, omega_0, omega_d, A)
        
        result = integrate.solve_ivp(
            fun=lfun,
            t_span=(0, tf),
            y0=y0,
            method="RK45",
            t_eval=t_steady
        )
        
        x = result.y[0]
        
        amplitude = (max(x) - min(x)) / 2
        amplitudes.append(amplitude)
    
    plt.figure(figsize=(10, 6))
    plt.plot(driving_freq, amplitudes, 'b-', linewidth=2)
    
    plt.title("Resonance Curve: Steady-State Amplitude vs Driving Frequency")
    plt.xlabel("Driving Frequency ($\\omega_d$)")
    plt.ylabel("Steady-State Amplitude")
    plt.grid(False)
    
    plt.axvline(x=omega_0, color='r', linestyle='--', label=f'Natural Freq ($\\omega_0$ = {omega_0})')
    plt.legend()
    
    plt.show()


if __name__ == '__main__':
    main()