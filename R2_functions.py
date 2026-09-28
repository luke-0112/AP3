import numpy as np

def differential_rl(v, r, l, i):
    
    didt = (v - r * i) / l
    return didt

def exact_solution_rl(v, r, l, t):
    
    i_exact = (v / r) * (1 - np.exp(-(r * t) / l))
    return i_exact