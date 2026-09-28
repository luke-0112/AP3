import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path


def simple_pendulum(t, y):
    x, v = y  #extract x and v values from tuple
    dydt = np.array([v, -x])  #generate an array with rates of change
    return dydt


def generate_path(home_folder=str(Path.home()), subfolder='/GitHub/', basename='output', extension='txt'):
    output_folder = home_folder + subfolder  
    filename = basename + '.' + extension  
    output_path = output_folder + filename  
    return output_path


def phase_space(x, v):
    plt.plot(x, v, 'k', linewidth=1.5)
    plt.axis('equal')

    # labels the axes using LaTeX formatting for nice math symbols
    plt.xlabel(r"x")
    plt.ylabel(r"v")
    plt.title("Phase Space")


def main():
    x0 = 0  #initial parameters
    v0 = 1
    y0 = (x0, v0)
    t0 = 0

    tf = 10 * np.pi
    n = 1001
    
    t = np.linspace(t0, tf, n)

    result = integrate.solve_ivp(
        fun=simple_pendulum,
        t_span=(t0, tf),
        y0=y0,
        method="RK45",
        t_eval=t
    )

    x, v = result.y
    t = result.t

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(t, x, label=r"x(t)")
    ax1.plot(t, v, label=r"v(t)")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("State (x, v)")
    ax1.set_title("Time Dependency")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.sca(ax2)
    phase_space(x, v)
    ax2.grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout()

    """filename = generate_path(basename='Harmonic-SHO-Output', extension='png')
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Output file saved to {filename}.")"""
    
    plt.show()

if __name__ == '__main__':
    main()