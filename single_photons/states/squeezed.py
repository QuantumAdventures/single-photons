import numpy as np
from single_photons.gaussian_state import gaussian_state


def squeezed(r=1):
    """Returns a squeezed state with real squeezing parameter r"""
    assert np.isreal(r), "Unsupported imaginary amplitude for squeezed state"
    
    R = np.array([[0], [0]])                                                    # Mean quadratures  of a coherent state with complex amplitude alpha
    V = np.diag([np.exp(-2*r), np.exp(+2*r)]);                                  # Covariance matrix of a coherent state with complex amplitude alpha
    
    return gaussian_state(R, V)