import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from R7_function import driven_pendulum


def main():
    
    A = 1.0
    omega_0 = 1.0
    

    x0 = 0.0
    v0 = 0.0
    y0 = (x0, v0)
    
    t0 = 0
    tf = 100.0
    n = 500
    
    damping_coefficients = [0.05, 0.1, 0.2, 0.5, 1.0]
    

    driving_frequencies = np.linspace(0.2 * omega_0, 2.0 * omega_0, 100)
    

    t_steady = np.linspace(0.8 * tf, tf, n)
    

    plt.figure(figsize=(10, 6))
    

    for b in damping_coefficients:
        
        amplitudes = []
        
        print(f"Running simulation for b = {b}...")
        
        for omega_d in driving_frequencies:
            
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
        
        plt.plot(driving_frequencies, amplitudes, label=f'b={b}', linewidth=2)
    
    plt.title("Driving Frequency Scan: Resonance Curves for Different Damping")
    plt.xlabel("Driving Frequency ($\\omega_d$ / $\\omega_0$)")
    plt.ylabel("Steady-State Amplitude")
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.xlim(0, 2.0)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()