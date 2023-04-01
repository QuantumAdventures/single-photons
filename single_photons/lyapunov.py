import numpy as np
from numpy.linalg import det
from numpy.linalg import matrix_power
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power


def lyapunov_ode_unconditional(t, V_old_vector, A, D):
    """
    Auxiliar internal function defining the Lyapunov equation 
    and calculating the derivative of the covariance matrix
    """
    
    M = A.shape[0];                                                             # System dimension (N_particles + 1 cavity field)partículas  + 1 campo)
    
    A_T = np.transpose(A)                                                       # Transpose of the drift matrix
    
    V_old = np.reshape(V_old_vector, (M, M));                                      # Vector -> matrix
    
    dVdt = np.matmul(A, V_old) + np.matmul(V_old, A_T) + D;                     # Calculate how much the CM derivative in this time step

    dVdt_vector = np.reshape(dVdt, (M**2,));                                     # Matrix -> vector
    return dVdt_vector

def lyapunov_ode_conditional(t, V_old_vector, A, D, B):
    """
    Auxiliar internal function defining the Lyapunov equation 
    and calculating the derivative of the covariance matrix
    """
    
    M = A.shape[0];                                                             # System dimension (N_particles + 1 cavity field)partículas  + 1 campo)
    
    A_T = np.transpose(A)                                                       # Transpose of the drift matrix
    
    V_old = np.reshape(V_old_vector, (M, M));                                   # Vector -> matrix
    
    # chi = np.matmul(C, V_old) + Gamma             # THIS SHOULD BE FASTER!
    # chi = np.matmul(np.transpose(chi), chi)
    # chi = np.matmul( np.matmul(V_old, np.transpose(C)) + np.transpose(Gamma),  np.matmul(C, V_old) + Gamma )    # Auxiliar matrix
    chi = np.matmul( np.matmul( np.matmul(V_old, B), np.transpose(B)), V_old)    # Auxiliar matrix
    dVdt = np.matmul(A, V_old) + np.matmul(V_old, A_T) + D - chi;               # Calculate how much the CM derivative in this time step
    
    dVdt_vector = np.reshape(dVdt, (M**2,));                                    # Matrix -> vector
    return dVdt_vector
