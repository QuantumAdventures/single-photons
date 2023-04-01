import numpy as np
from single_photons.gaussian_state import gaussian_state

def vacuum(N=1):
    """Returns an N-mode tensor product of vacuum states. Default N=1"""
    
    R = np.zeros(2*N)
    V = np.eye(2*N)
    
    return gaussian_state(R, V)