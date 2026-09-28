import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path

#import function
from R4_function import damped_pendulum

def generate_path(home_folder=str(Path.home()), subfolder='/Documents/', basename='output', extension='png'):
    output_folder = home_folder + subfolder  
    filename = basename + '.' + extension  
    output_path = output_folder + filename  
    return output_path

def main():
    
    regimes = [
        {'b': 0.5, 'label': 'Underdamped (b=0.5)'},
        {'b': 2.0, 'label': 'Critically Damped (b=2.0)'},
        {'b': 5.0, 'label': 'Overdamped (b=5.0)'}
    ]
    
    omega0 = 1.0
    x0 = 1.0   #initial displacement
    v0 = 0.0   #initial velocity is zero
    y0 = (x0, v0)
    
    t0 = 0
    tf = 20.0
    n = 1000
    t_eval = np.linspace(t0, tf, n)
    
    # Increased figsize height from 12 to 14 to give more room
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    for i, regime in enumerate(regimes):
        b_val = regime['b']
        label = regime['label']
        
        #solve
        result = integrate.solve_ivp(
            fun=lambda t, y: damped_pendulum(t, y, b=b_val, omega0=omega0),
            t_span=(t0, tf),
            y0=y0,
            method="RK45",
            t_eval=t_eval
        )
        
        x, v = result.y
        t = result.t
        
        #plot L
        ax_time = axes[i, 0]
        ax_time.plot(t, x, label="x(t)")
        ax_time.plot(t, v, label="v(t)")
        ax_time.set_title(f"Time Dependency: {label}")
        ax_time.set_xlabel("Time (t)")
        ax_time.set_ylabel("State (x, v)")
        ax_time.legend(loc='upper right')
        ax_time.grid(False)
        
        #plot R
        ax_phase = axes[i, 1]
        ax_phase.plot(x, v, 'k', linewidth=1.5)
        ax_phase.set_title(f"Phase Space: {label}")
        ax_phase.set_xlabel("x")
        ax_phase.set_ylabel("v")
        ax_phase.grid(False)

        #at rest => maximum amplitude < initial(0.01)
        max_amp = np.max(np.abs(x))
        if max_amp < 0.01:
            print(f"{label}: Came to rest (amplitude < 0.01) within {tf} seconds.")
        else:
            print(f"{label}: Still oscillating at t={tf}s (max amplitude = {max_amp:.4f})")

    plt.subplots_adjust(hspace=0.5)
    
    """
    filename = generate_path(basename='Damped-Oscillator-Comparison', extension='png')
    plt.savefig(filename, bbox_inches='tight')
    print(f"Output file saved to {filename}.")"""
    
    plt.show()

if __name__ == '__main__':
    main()