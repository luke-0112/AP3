import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define the nonlinear derivative function
# We add 'a' and 'b' as arguments so we can change them easily later
def nonlinear1(t, y, a, b):
    dydt = -a * y**3 + b * np.sin(t)
    return dydt

def main():
    # Initial conditions
    y0 = np.array([0])  # initial state at t = 0
    t0 = 0              # initial time
    tf = 20             # final time
    n = 101             # Number of points (100 steps)

    # Create a numpy array of n times linearly spaced between t0 and tf
    t = np.linspace(t0, tf, n)

    # Create a figure and axis object for better control
    plt.figure(figsize=(10, 6))
    
    # Define a list of parameter sets to test
    # Format: (a_value, b_value, color, label_string)
    # I chose a few variations to show how the curve changes
    parameters = [
        (1.0, 1.0, 'blue', 'a=1, b=1'),
        (0.5, 1.0, 'red', 'a=0.5, b=1'),
        (2.0, 1.0, 'green', 'a=2, b=1'),
        (1.0, 0.5, 'orange', 'a=1, b=0.5'),
        (1.0, 2.0, 'purple', 'a=1, b=2')
    ]

    # Loop through each parameter set and solve the ODE
    for a_val, b_val, color, label in parameters:
        
        # Call the RK integrator
        # We pass 'args=(a_val, b_val)' to pass the parameters to nonlinear1
        result = integrate.solve_ivp(
            fun=nonlinear1, 
            t_span=(t0, tf), 
            y0=y0, 
            method="RK45", 
            t_eval=t,
            args=(a_val, b_val) 
        )

        # Read the solution and time
        y = result.y[0]
        t_sol = result.t

        # Plot the solution with labels for the legend 
        plt.plot(t_sol, y, '.', color=color, label=label, markersize=8)

    plt.title('Solutions of dy/dt = -ay³ + bsin(t) for Various Parameters')
    plt.xlabel('Time (t)')
    plt.ylabel('Displacement (y)')
    plt.legend() # Show the legend
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()