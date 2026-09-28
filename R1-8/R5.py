import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path

#import
from R4_function import damped_pendulum

def generate_path(home_folder=str(Path.home()), subfolder='/Documents/', basename='output', extension='png'):
    output_folder = home_folder + subfolder  
    filename = basename + '.' + extension  
    output_path = output_folder + filename  
    return output_path

def main():
    
    b = 0.8
    omega0 = 2.0
    
    lfun = lambda t, y: damped_pendulum(t, y, b, omega0)
    
    x0 = 1.0
    v0 = 0.0
    y0 = (x0, v0)
    t0 = 0
    tf = 20.0
    n = 1000
    t = np.linspace(t0, tf, n)  
    
    result = integrate.solve_ivp(
        fun=lfun,
        t_span=(t0, tf),
        y0=y0,
        method="RK45",
        t_eval=t
    )
    
    x, v = result.y
    t_sol = result.t
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax_time = axes[0]
    ax_time.plot(t_sol, x, label="x(t)")
    ax_time.plot(t_sol, v, label="v(t)")
    ax_time.set_title(f"Time Dependency: b={b}, $\omega_0$={omega0}")
    ax_time.set_xlabel("Time (t)")
    ax_time.set_ylabel("State (x, v)")
    ax_time.legend(loc='upper right')
    ax_time.grid(False)
    
    ax_phase = axes[1]
    ax_phase.plot(x, v, 'k', linewidth=1.5)
    ax_phase.set_title(f"Phase Space: b={b}, $\omega_0$={omega0}")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("v")
    ax_phase.grid(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()