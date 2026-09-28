import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from R6_function import driven_pendulum


def main():
    
    A = 1.0
    b = 0.1
    omega0 = 1.0

    x0 = 0.0
    v0 = 0.0
    y0 = (x0, v0)
    
    t0 = 0
    tf = 100.0
    n = 5000
    t = np.linspace(t0, tf, n)
    

    omegad_values = (omega0, 0.9 * omega0, 0.5 * omega0)
    
    plt.figure(figsize=(10, 6))
    
    results = []
    
    for omegad in omegad_values:

        lfun = lambda t, y: driven_pendulum(t, y, b, omega0, omegad, A)
        
        result = integrate.solve_ivp(
            fun=lfun,
            t_span=(t0, tf),
            y0=y0,
            method="RK45",
            t_eval=t
        )
        
        t_sol = result.t
        x, v = result.y
        
        results.append((omegad, t_sol, x, v))
        
        plt.plot(t_sol, x, label=f'x(t): $\omega_d$ = {omegad:.2f}')
    
    plt.title("Driven Oscillator: Response for Different Driving Frequencies")
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement (x)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)
    
    for i, (omegad, t_sol, x, v) in enumerate(results):
        plt.subplot(1, 3, i+1)
        plt.plot(x, v, 'k', linewidth=1.0)
        plt.axis('equal')
        plt.title(f"Phase Space: $\omega_d$ = {omegad:.2f}")
        plt.xlabel("x")
        plt.ylabel("v")
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()


if __name__ == '__main__':
    main()