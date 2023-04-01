import numpy as np
from single_photons.gaussian_state import gaussian_state

def thermal(nbar=1):
    """Returns a thermal state with mean occupation number nbar"""
    assert nbar>=0, "Imaginary or negative occupation number for thermal state"
    
    R = np.array([[0], [0]])                                                    # Mean quadratures  of a coherent state with complex amplitude alpha
    V = np.diag([2.0*nbar+1, 2.0*nbar+1]);                                      # Covariance matrix of a coherent state with complex amplitude alpha  
    
    return gaussian_state(R, V)