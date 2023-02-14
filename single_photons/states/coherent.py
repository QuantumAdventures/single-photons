import numpy as np
from single_photons import gaussian_state

def coherent(alpha=1):
    """Returns a coherent state with complex amplitude alpha"""
    R = np.array([[2*alpha.real], [2*alpha.imag]]);                             # Mean quadratures  of a coherent state with complex amplitude alpha
    V = np.identity(2);                                                         # Covariance matrix of a coherent state with complex amplitude alpha
    
    return gaussian_state(R, V)