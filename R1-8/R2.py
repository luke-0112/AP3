import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#import functions 
from R2_functions import differential_rl, exact_solution_rl


def main():
    V = 10.0    #voltage
    R = 50.0    #resistance
    L = 100.0   #inductance
    I0 = 0.0    #initial current

    t0 = 0.0
    tf = 10.0
    n = 100
    
    #time array for ODE solver
    t_eval = np.linspace(t0, tf, n)
    t_span = (t0, tf)
    
    result = integrate.solve_ivp(
        fun=lambda t, i: differential_rl(V, R, L, i),
        t_span=t_span,
        y0=[I0],
        method="RK45",
        t_eval=t_eval
    )

    t_numeric = result.t
    i_numeric = result.y[0]
    i_exact = exact_solution_rl(V, R, L, t_numeric)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_numeric, i_numeric, 'bo', label='Numerical (RK45)', markersize=5)
    plt.plot(t_numeric, i_exact, 'r-', label='Exact Analytical Solution', linewidth=2)

    plt.title('RL Circuit Transient Response: Numerical vs Exact')
    plt.xlabel('Time (t) / s')
    plt.ylabel('Current (I) / A')
    plt.grid(False)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()