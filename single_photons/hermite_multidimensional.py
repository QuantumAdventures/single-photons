import numpy as np
from numpy.linalg import det
from numpy.linalg import matrix_power
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power


def Hermite_multidimensional_original(R, y, n_cutoff=10):
    """
    Calculates the multidimensional Hermite polynomial H_m^R(y) from m = (0, ..., 0) up to (n_cutoff, ..., n_cutoff)
    
    ARGUMENTS:
        R - n by n symmetric matrix
        y - n-dimensional point where the polynomial is to be evaluated
        n_cutoff - maximum value for the polynomial to be calculated
        
    RETURNS:
        H - tensor with the multidimensional Hermite polynomial evaluated on n-dimensional grid ( H.shape = n*[n_cutoff] )
    
    REFERENCE:
        Math. Comp. 24, 537-545 (1970)
    """
    n = len(R)                                                                  # Dimension of the input matrix and vector
    
    H = np.zeros(n*[n_cutoff], dtype=complex)                                   # Initialize the tensor to 0 (n_cutoff entries in each of the dim dimensions)

    m_last = np.array(n*[0], dtype=int)                                         # Linear index for last altered entry of the Hermite tensor
    m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')       # Get "linearized" index (Adjust from Python indexig to original article indexing starting a 1)
    H.ravel()[m_last_linear] = 1                                                # Set its first entry to 1 (known value)
    
    n_entries = np.prod(H.shape)                                                # Number of entries on the final tensor
    
    for m_next_linear in range(1, n_entries):                                 # Loop through every entry on tensor H using linear indices ( m is the linearized index for H: H.ravel()[m] <-> H[ np.unravel_index(m, H.shape, order='F') ] )
        
        m_next = np.array(np.unravel_index(m_next_linear, H.shape, order='F'), dtype=int)  # Vector index for the next entry of the Hermite tensor to be calculated
        
        e_k = m_next - m_last                                                   # Discover how much it has transversed since last iteration
        
        if np.any(e_k<0):                                                       # If it changed the dimension transversed
            m_last[e_k<0] = 0                                                   # Move the last position accordingly
            m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F') # Update the last linear index accordingly
            
            e_k[e_k<0] = 0                                                      # Remember to alter notation (e_k must be only zeros and a single one)
        
        k = np.where(e_k.squeeze())[0]                                          # Index for the entry where e_k == 1 (should be only one entry that matches this condition!)
        
        # Calculate the first term of this new entry
        R_times_y = 0
        for j in range(n):                                                      # This loop is essentially the sum on this first term
            R_times_y = R_times_y + R[k,j]*y[j,0]
            
        H.ravel()[m_next_linear] = R_times_y*H.ravel()[m_last_linear]           # Remember that m_last = m_next - e_k
        
        #  Calculate the second term of this new entry
        for j in range(n):
            e_j = np.zeros(n, dtype=int)
            e_j[j] = 1                                                          # For this j, build the vector e_j
            
            m_jk = m_last - e_j
            if (j == k) or np.any(m_jk < 0):                                    # If you end up with a negative index 
                continue                                                        # the corresponding entry of the tensor is null
            
            m_jk_linear = np.ravel_multi_index(m_jk, dims=H.shape, order='F')
            H.ravel()[m_next_linear] = H.ravel()[m_next_linear] - m_next[j]*R[k,j]*H.ravel()[m_jk_linear]
            
        #  Calculate the last term of this new entry
        m_2k = m_next - 2*e_k
        if np.all(m_2k >= 0):
            m_2k_linear = np.ravel_multi_index(m_2k, dims=H.shape, order='F')
            H.ravel()[m_next_linear] =  H.ravel()[m_next_linear] - R[k,k]*(m_next[k]-1)*H.ravel()[m_2k_linear]
        
        # Update the last index before moving to the next iteration
        m_last = m_next                                                         # Update the last vector index of the loop
        m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')   # Update the last linear index of the loop

    H = H.real                                                                  # Get rid off any residual complex value
    
    return H

def Hermite_multidimensional(R, y, n_cutoff=10):
    """
    Calculates the multidimensional Hermite polynomial H_m^R(y) from m = (0, ..., 0) up to (n_cutoff, ..., n_cutoff)
    
    ARGUMENTS:
        R - n by n symmetric matrix
        y - n-dimensional point where the polynomial is to be evaluated
        n_cutoff - maximum value for the polynomial to be calculated
        
    RETURNS:
        H - tensor with the multidimensional Hermite polynomial evaluated on n-dimensional grid ( H.shape = n*[n_cutoff] )
    
    REFERENCE:
        Math. Comp. 24, 537-545 (1970)
    """
    n = len(R)                                                                  # Dimension of the input matrix and vector
    
    H = np.zeros(n*[n_cutoff], dtype=complex)                                   # Initialize the tensor to 0 (n_cutoff entries in each of the dim dimensions)

    m_last = np.array(n*[0], dtype=int)                                         # Linear index for last altered entry of the Hermite tensor
    m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')       # Get "linearized" index (Adjust from Python indexig to original article indexing starting a 1)
    H.ravel()[m_last_linear] = 1                                                # Set its first entry to 1 (known value)
    
    n_entries = np.prod(H.shape)                                                # Number of entries on the final tensor
    
    for m_next_linear in range(1, n_entries):                                 # Loop through every entry on tensor H using linear indices ( m is the linearized index for H: H.ravel()[m] <-> H[ np.unravel_index(m, H.shape, order='F') ] )
        
        m_next = np.array(np.unravel_index(m_next_linear, H.shape, order='F'), dtype=int)  # Vector index for the next entry of the Hermite tensor to be calculated
        
        e_k = m_next - m_last                                                   # Discover how much it has transversed since last iteration
        
        if np.any(e_k<0):                                                       # If it changed the dimension transversed
            m_last[e_k<0] = 0                                                   # Move the last position accordingly
            m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F') # Update the last linear index accordingly
            
            e_k[e_k<0] = 0                                                      # Remember to alter notation (e_k must be only zeros and a single one)
        
        k = np.where(e_k.squeeze())[0]                                          # Index for the entry where e_k == 1 (should be only one entry that matches this condition!)
        
        # Debugging
        # if np.any(m_last<0):
        #     a=1
        
        # Calculate the first term of this new entry
        R_times_y = 0
        for j in range(n):                                                      # This loop is essentially the sum on this first term
            R_times_y = R_times_y + R[k,j]*y[j,0]
            
        H.ravel()[m_next_linear] = R_times_y*H.ravel()[m_last_linear]           # Remember that m_last = m_next - e_k
        
        #  Calculate the second term of this new entry
        for j in range(n):
            e_j = np.zeros(n, dtype=int)
            e_j[j] = 1                                                          # For this j, build the vector e_j
            
            m_jk = m_last - e_j
            if (j==k) or np.any(m_jk < 0):                                                # If you end up with a negative index 
                continue                                                        # the corresponding entry of the tensor is null
            
            m_jk_linear = np.ravel_multi_index(m_jk, dims=H.shape, order='F')
            H.ravel()[m_next_linear] = H.ravel()[m_next_linear] - m_next[j]*R[k,j]*H.ravel()[m_jk_linear]
            
        #  Calculate the last term of this new entry
        m_2k = m_next - 2*e_k
        if np.all(m_2k >= 0):
            m_2k_linear = np.ravel_multi_index(m_2k, dims=H.shape, order='F')
            H.ravel()[m_next_linear] =  H.ravel()[m_next_linear] - R[k,k]*(m_next[k]-1)*H.ravel()[m_2k_linear]
        
        # Update the last index before moving to the next iteration
        m_last = m_next                                                         # Update the last vector index of the loop
        m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')   # Update the last linear index of the loop

    H = H.real                                                                  # Get rid off any residual complex value
    
    return H