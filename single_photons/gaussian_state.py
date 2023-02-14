# -*- coding: utf-8 -*-
"""
QuGIT - Quantum Gaussian Information Toolbox
Github: https://github.com/IgorBrandao42/Quantum-Gaussian-Information-Toolbox

Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""


import numpy as np
from numpy.linalg import det
from numpy.linalg import matrix_power

from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power


################################################################################


class gaussian_state:                                                           # Class definning a multimode gaussian state
    """Class simulation of a multimode gaussian state
    
    ATTRIBUTES:
        self.R       - Mean quadratures vector
        self.V       - Covariance matrix
        self.Omega   - Symplectic form matrix
        self.N_modes - Number of modes
    """    
    
    # Constructor and its auxiliar functions    
    def __init__(self, *args):
        """
        The user can explicitly pass the first two moments of a multimode gaussian state
        or pass a name-value pair argument to choose a single mode gaussian state
        
        PARAMETERS:
            R0, V0 - mean quadratures vector and covariance matrix of a gaussian state (ndarrays)
            
        NAME-VALUE PAIR ARGUMENTS:
            "vacuum"                        - generates vacuum   state (string)
            "thermal" , occupation number   - generates thermal  state (string, float)
            "coherent", complex amplitude   - generates coherent state (string, complex)
            "squeezed", squeezing parameter - generates squeezed state (string, float)
        """

        if(len(args) == 0):                                                     # Default constructor (vacuum state)
            self.R = np.array([[0], [0]])                                       # Save mean quadratres   in a class attribute
            self.V = np.identity(2)                                             # Save covariance matrix in a class attribute
            self.N_modes = 1;
             
        elif( isinstance(args[0], str) ):                                       # If the user called for an elementary gaussian state
            self.decide_which_state(args)                                       # Call the proper method to decipher which state the user wants 
        
        elif(isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray)): # If the user gave the desired mean quadratures values and covariance matrix
            R0 = args[0]
            V0 = args[1]
            
            R_is_real = all(np.isreal(R0))
            R_is_vector = np.squeeze(R0).ndim == 1
            
            V_is_matrix = np.squeeze(V0).ndim == 2
            V_is_square = V0.shape[0] == V0.shape[1]
            
            R_and_V_match = len(R0) == len(V0)
            
            assert R_is_real and R_is_vector and V_is_matrix and R_and_V_match and V_is_square, "Unexpected first moments when creating gaussian state!"  # Make sure they are a vector and a matrix with same length
        
            self.R = np.vstack(R0);                                             # Save mean quadratres   in a class attribute (vstack to ensure column vector)
            self.V = V0;                                                        # Save covariance matrix in a class attribute
            self.N_modes = int(len(R0)/2);                                           # Save the number of modes of the multimode state in a class attribute
            
        else:
            raise ValueError('Unexpected arguments when creating gaussian state!') # If input arguments do not make sense, call out the user
        
        omega = np.array([[0, 1], [-1, 0]]);                                    # Auxiliar variable
        self.Omega = np.kron(np.eye(self.N_modes,dtype=int), omega)             # Save the symplectic form matrix in a class attribute                                                    
    
    def decide_which_state(self, varargin):
        # If the user provided a name-pair argument to the constructor,
        # this function reads these arguments and creates the first moments of the gaussian state
      
        self.N_modes = 1;
        type_state = varargin[0];                                               # Name of expected type of gaussian state
      
        if(str(type_state) == "vacuum"):                                        # If it is a vacuum state
            self.R = np.array([[0], [0]])                                       # Save mean quadratres   in a class attribute
            self.V = np.identity(2)                                             # Save covariance matrix in a class attribute
            self.N_modes = 1;
            return                                                              # End function
      
                                                                                # Make sure there is an extra parameters that is a number
        assert len(varargin)>1, "Absent amplitude for non-vacuum elementary gaussian state"
        assert isinstance(varargin[1], (int, float, complex)), "Invalid amplitude for non-vacuum elementary gaussian state"
        
        if(str(type_state) == "thermal"):                                       # If it is a thermal state
            nbar = varargin[1];                                                 # Make sure its occuption number is a non-negative number
            assert nbar>=0, "Imaginary or negative occupation number for thermal state"
            self.R = np.array([[0], [0]])
            self.V = np.diag([2.0*nbar+1, 2.0*nbar+1]);                         # Create its first moments
        
        elif(str(type_state) == "coherent"):                                    # If it is a coherent state
           alpha = varargin[1]
           self.R = np.array([[2*alpha.real], [2*alpha.imag]])
           self.V = np.identity(2)                                             # Create its first moments
        
        elif(str(type_state) == "squeezed"):                                    # If it is a squeezed state
            r = varargin[1]                                                    # Make sure its squeezing parameter is a real number
            assert np.isreal(r), "Unsupported imaginary amplitude for squeezed state"
            self.R = np.array([[0], [0]])
            self.V = np.diag([np.exp(-2*r), np.exp(+2*r)])                           # Create its first moments
        
        else:
            self.N_modes = []
            raise ValueError("Unrecognized gaussian state name, please check for typos or explicitely pass its first moments as arguments")
    
    def check_uncertainty_relation(self):
      """
      Check if the generated covariance matrix indeed satisfies the uncertainty principle (debbugging)
      """
      
      V_check = self.V + 1j*self.Omega;
      eigvalue, eigvector = np.linalg.eig(V_check)
      
      assert all(eigvalue>=0), "CM does not satisfy uncertainty relation!"
      
      return V_check
    
    def __str__(self):
        return str(self.N_modes) + "-mode gaussian state with mean quadrature vector R =\n" + str(self.R) + "\nand covariance matrix V =\n" + str(self.V)
    
    def copy(self):
        """Create identical copy"""
        
        return gaussian_state(self.R, self.V)
    
    # Construct another state, from this base gaussian_state
    def tensor_product(self, rho_list):
        """ Given a list of gaussian states, 
        # calculates the tensor product of the base state and the states in the array
        # 
        # PARAMETERS:
        #    rho_array - array of gaussian_state (multimodes)
        #
         CALCULATES:
            rho - multimode gaussian_state with all of the input states
        """
      
        R_final = self.R;                                                      # First moments of resulting state is the same of rho_A
        V_final = self.V;                                                      # First block diagonal entry is the CM of rho_A
      
        for rho in rho_list:                                                    # Loop through each state that is to appended
            R_final = np.vstack((R_final, rho.R))                               # Create its first moments
            V_final = block_diag(V_final, rho.V)
        
        temp = gaussian_state(R_final, V_final)                                 # Generate the gaussian state with these moments
        
        self.R = temp.R                                                         # Copy its attributes into the original instance
        self.V = temp.V
        self.Omega   = temp.Omega
        self.N_modes = temp.N_modes
    
    def partial_trace(self, indexes):
        """
        Partial trace over specific single modes of the complete gaussian state
        
        PARAMETERS:
           indexes - the modes the user wants to trace out (as in the mathematical notation) 
        
        CALCULATES:
           rho_A - gaussian_state with all of the input state, except of the modes specified in 'indexes'
        """
      
        N_A = int(len(self.R) - 2*len(indexes));                                    # Twice the number of modes in resulting state
        assert N_A>=0, "Partial trace over more states than there exist in gaussian state" 
      
        # Shouldn't there be an assert over max(indexes) < obj.N_modes ? -> you cant trace out modes that do not exist
      
        modes = np.arange(self.N_modes)
        entries = np.isin(modes, indexes)
        entries = [not elem for elem in entries]
        modes = modes[entries];
      
        R0 = np.zeros((N_A, 1))
        V0 = np.zeros((N_A,N_A))
      
        for i in range(len(modes)):
            m = modes[i]
            R0[(2*i):(2*i+2)] = self.R[(2*m):(2*m+2)]
        
            for j in range(len(modes)):
                n = modes[j]
                V0[(2*i):(2*i+2), (2*j):(2*j+2)] = self.V[(2*m):(2*m+2), (2*n):(2*n+2)]
        
        temp = gaussian_state(R0, V0)                                          # Generate the gaussian state with these moments
        
        self.R = temp.R                                                         # Copy its attributes into the original instance
        self.V = temp.V
        self.Omega   = temp.Omega
        self.N_modes = temp.N_modes
    
    def only_modes(self, indexes):
      """
      Partial trace over all modes except the ones in indexes of the complete gaussian state
       
       PARAMETERS:
          indexes - the modes the user wants to retrieve from the multimode gaussian state
      
       CALCULATES:
          rho - gaussian_state with all of the specified modes
      """
      
      N_A = len(indexes);                                                       # Number of modes in resulting state
      assert N_A>0 and N_A <= self.N_modes, "Partial trace over more states than exists in gaussian state"
      
      R0 = np.zeros((2*N_A, 1))
      V0 = np.zeros((2*N_A, 2*N_A))
      
      for i in range(len(indexes)):
            m = indexes[i]
            R0[(2*i):(2*i+2)] = self.R[(2*m):(2*m+2)]
        
            for j in range(len(indexes)):
                n = indexes[j]
                V0[(2*i):(2*i+2), (2*j):(2*j+2)] = self.V[(2*m):(2*m+2), (2*n):(2*n+2)]
      
      temp = gaussian_state(R0, V0);                                            # Generate the gaussian state with these moments
        
      self.R = temp.R                                                           # Copy its attributes into the original instance
      self.V = temp.V
      self.Omega   = temp.Omega
      self.N_modes = temp.N_modes  
    
    def loss_ancilla(self,idx,tau):
        """
        Simulates a generic loss on mode idx by anexing an ancilla vacuum state and applying a
        beam splitter operator with transmissivity tau. The ancilla is traced-off from the final state. 
        
        PARAMETERS:
           idx - index of the mode that will suffer loss
           tau - transmissivity of the beam splitter
        
        CALCULATES:
            damped_state - final damped state
        """

        damped_state = tensor_product([self, gaussian_state("vacuum")])
        damped_state.beam_splitter(tau,[idx, damped_state.N_modes-1])
        damped_state.partial_trace([damped_state.N_modes-1])
        
        self.R = damped_state.R                                                 # Copy the damped state's attributes into the original instance
        self.V = damped_state.V
        self.Omega   = damped_state.Omega
        self.N_modes = damped_state.N_modes
    
    # Properties of the gaussian state
    def symplectic_eigenvalues(self):
        """
        Calculates the sympletic eigenvalues of a covariance matrix V with symplectic form Omega
        
        Finds the absolute values ofthe eigenvalues of i\Omega V and removes repeated entries
        
        CALCULATES:
            lambda - array with symplectic eigenvalues
        """  
        H = 1j*np.matmul(self.Omega, self.V)                                   # Auxiliar matrix
        lambda_0, v_0 = np.linalg.eig(H)
        lambda_0 = np.abs( lambda_0 )                                       # Absolute value of the eigenvalues of the auxiliar matrix
        
        lambda_s = np.zeros((self.N_modes, 1));                                 # Variable to store the symplectic eigenvalues
        for i in range(self.N_modes):                                           # Loop over the non-repeated entries of lambda_0
            lambda_s[i] = lambda_0[0]                                         # Get the first value on the repeated array
            lambda_0 = np.delete(lambda_0, 0)                                  # Delete it
            
            idx = np.argmin( np.abs(lambda_0-lambda_s[i]) )                           # Find the next closest value on the array (repeated entry)
            lambda_0 = np.delete(lambda_0, idx)                              # Delete it too
        
        return lambda_s
    
    def purity(self):
      """
      Purity of a gaussian state (pure states have unitary purity)
       
       CALCULATES:
           p - purity
      """
      
      return 1/np.prod( self.symplectic_eigenvalues() );
    
    def squeezing_degree(self):
        """
        Degree of squeezing of the quadratures of a single mode state
        Defined as the ratio of the variance of the squeezed and antisqueezed quadratures
        
        CALCULATES:
            eta   - ratio of the variances above
            V_sq  - variance of the     squeezed quadrature
            V_asq - variance of the antisqueezed quadrature
                   
        REFERENCE: 
            Phys. Rev. Research 2, 013052 (2020)
        """
      
        assert self.N_modes == 1, "At the moment, this program only calculates the squeezing degree for a single mode state"
      
        lambda_0, v_0 = np.linalg.eig(self.V)
        
        V_sq  = np.amin(lambda_0)
        V_asq = np.amax(lambda_0)
      
        eta = V_sq/V_asq;
        return eta, V_sq, V_asq
    
    def von_Neumann_Entropy(self):
        """
        Calculation of the von Neumann entropy for a multipartite gaussian system
       
        CALCULATES:
             Entropy - von Neumann entropy of the multimode state
        """
        
        nu = self.symplectic_eigenvalues();                                     # Calculates the sympletic eigenvalues of a covariance matrix V
        
                                                                                # 0*log(0) is NaN, but in the limit that x->0 : x*log(x) -> 0
        # nu[np.abs(nu - 1) < 1e-15] = nu[np.abs(nu - 1) < 1e-15] + 1e-15;                                 # Doubles uses a 15 digits precision, I'm adding a noise at the limit of the numerical precision
        nu[np.abs(nu-1) < 1e-15] = 1+1e-15
        
        nu_plus  = (nu + 1)/2.0;                                                # Temporary variables
        # nu_minus = (nu - 1)/2.0;
        nu_minus = np.abs((nu - 1)/2.0);
        g_nu = np.multiply(nu_plus,np.log(nu_plus)) - np.multiply(nu_minus, np.log(nu_minus))
      
        Entropy = np.sum( g_nu );                                               # Calculate the entropy
        return Entropy
    
    def mutual_information(self):
        """
         Mutual information for a multipartite gaussian system
        
         CALCULATES:
            I     - mutual information  for the total system of the j-th covariance matrix
            S_tot - von Neumann entropy for the total system of the j-th covariance matrix
            S     - von Neumann entropy for the i-th mode    of the j-th covariance matrix
        """
        S = np.zeros((self.N_modes, 1));                                        # Variable to store the entropy of each mode
        
        for j in range(self.N_modes):                                           # Loop through each mode
            single_mode = only_modes(self, [j]);                                # Get the covariance matrix for only the i-th mode
            S[j] = single_mode.von_Neumann_Entropy();                           # von Neumann Entropy for i-th mode of each covariance matrix
        
        S_tot = self.von_Neumann_Entropy();                                     # von Neumann Entropy for the total system of each covariance matrix
        
        I = np.sum(S) - S_tot;                                                  # Calculation of the mutual information
        return I
    
    def occupation_number(self):
        """
        Occupation number for a each single mode within the multipartite gaussian state (array)
        
        CALCULATES:
            nbar - array with the occupation number for each single mode of the multipartite gaussian state
        """
        
        Variances = np.diag(self.V);                                                # From the current CM, take take the variances
        Variances = np.vstack(Variances)
        
        mean_x = self.R[::2];                                                    # Odd  entries are the mean values of the position
        mean_p = self.R[1::2];                                                   # Even entries are the mean values of the momentum
        
        Var_x = Variances[::2];                                                 # Odd  entries are position variances
        Var_p = Variances[1::2];                                                # Even entries are momentum variances
        
        nbar = 0.25*( Var_x + mean_x**2 + Var_p + mean_p**2 ) - 0.5;            # Calculate occupantion numbers at current time
        return nbar
    
    def number_operator_moments(self):
        """
        Calculates means vector and covariance matrix of photon numbers for each mode of the gaussian state
        
        CALCULATES:
            m - mean values of number operator in arranged in a vector (Nx1 numpy.ndarray)
            K - covariance matrix of the number operator               (NxN numpy.ndarray)
           
        REFERENCE:
            Phys. Rev. A 99, 023817 (2019)
            Many thanks to Daniel Tandeitnik for the base code for this method!
        """
        q = self.R[::2]                                                         # Mean values of position quadratures (even entries of self.R)
        p = self.R[1::2]                                                        # Mean values of momentum quadratures (odd  entries of self.R)
        
        alpha   = 0.5*(q + 1j*p)                                                # Mean values of annihilation operators
        alpha_c = 0.5*(q - 1j*p)                                                # Mean values of creation     operators
        
        V_1 = self.V[0::2, 0::2]/2.0                                            # Auxiliar matrix
        V_2 = self.V[0::2, 1::2]/2.0                                            # Auxiliar matrix
        V_3 = self.V[1::2, 1::2]/2.0                                            # Auxiliar matrix
        
        A = ( V_1 + V_3 + 1j*(np.transpose(V_2) - V_2) )/2.0                    # Auxiliar matrix
        B = ( V_1 - V_3 + 1j*(np.transpose(V_2) + V_2)   )/2.0                    # Auxiliar matrix
        
        temp = np.multiply(np.matmul(alpha_c, alpha.transpose()), A) + np.multiply(np.matmul(alpha_c, alpha_c.transpose()), B) # Yup, you guessed it, another auxiliar matrix
        
        m = np.real(np.reshape(np.diag(A), (self.N_modes,1)) + np.multiply(alpha, alpha_c) - 0.5) # Mean values of number operator (occupation numbers)
        
        K = np.real(np.multiply(A, A.conjugate()) + np.multiply(B, B.conjugate()) - 0.25*np.eye(self.N_modes)  + 2.0*temp.real) # Covariance matrix for the number operator
        
        return m, K
    
    def coherence(self):
        """
        Coherence of a multipartite gaussian system
         
        CALCULATES:
            C - coherence
        
        REFERENCE: 
            Phys. Rev. A 93, 032111 (2016).
        """
        
        nbar = self.occupation_number();                                        # Array with each single mode occupation number
        
        nbar[nbar==0] = nbar[nbar==0] + 1e-16;                                  # Make sure there is no problem with log(0)!
        
        S_total = self.von_Neumann_Entropy();                                    # von Neumann Entropy for the total system
        
        temp = np.sum( np.multiply(nbar+1, np.log2(nbar+1)) - np.multiply(nbar, np.log2(nbar)) );                # Temporary variable
        
        C = temp - S_total;                                                     # Calculation of the mutual information
        return C
    
    def logarithmic_negativity(self, *args):
        """
        Calculation of the logarithmic negativity for a bipartite system
       
        PARAMETERS:
           indexes - array with indices for the bipartition to consider 
           If the system is already bipartite, this parameter is optional !
       
        CALCULATES:
           LN - logarithmic negativity for the bipartition / bipartite states
        """
        
        temp = self.N_modes 
        if(temp == 2):                                                          # If the full system is only comprised of two modes
            V0 = self.V                                                         # Take its full covariance matrix
        elif(len(args) > 0 and temp > 2):
            indexes = args[0]
            
            assert len(indexes) == 2, "Can only calculate the logarithmic negativity for a bipartition!"
                
            bipartition = only_modes(self,indexes)                              # Otherwise, get only the two mode specified by the user
            V0 = bipartition.V                                                  # Take the full Covariance matrix of this subsystem
        else:
            raise TypeError('Unable to decide which bipartite entanglement to infer, please pass the indexes to the desired bipartition')
        
        A = V0[0:2, 0:2]                                                        # Make use of its submatrices
        B = V0[2:4, 2:4] 
        C = V0[0:2, 2:4] 
        
        sigma = np.linalg.det(A) + np.linalg.det(B) - 2.0*np.linalg.det(C)      # Auxiliar variable
        
        ni = sigma/2.0 - np.sqrt( sigma**2 - 4.0*np.linalg.det(V0) )/2.0 ;      # Square of the smallest of the symplectic eigenvalues of the partially transposed covariance matrix
        
        if(ni < 0.0):                                                           # Manually perform a maximum to save computational time (calculation of a sqrt can take too much time and deal with residual numeric imaginary parts)
            LN = 0.0;
        else:
            ni = np.sqrt( ni.real );                                            # Smallest of the symplectic eigenvalues of the partially transposed covariance matrix
        
        LN = np.max([0, -np.log(ni)]);                                          # Calculate the logarithmic negativity at each time
        return LN
    
    def fidelity(self, rho_2):
        """
        Calculates the fidelity between the two arbitrary gaussian states
        
        ARGUMENTS:
            rho_1, rho_2 - gaussian states to be compared through fidelity
         
        CALCULATES:
            F - fidelity between rho_1 and rho_2
        
        REFERENCE:
            Phys. Rev. Lett. 115, 260501.
       
        OBSERVATION:
        The user should note that non-normalized quadratures are expected;
        They are normalized to be in accordance with the notation of Phys. Rev. Lett. 115, 260501.
        """
      
        assert self.N_modes == rho_2.N_modes, "Impossible to calculate the fidelity between gaussian states of diferent sizes!" 
        
        u_1 = self.R/np.sqrt(2.0);                                              # Normalize the mean value of the quadratures
        u_2 = rho_2.R/np.sqrt(2.0);
        
        V_1 = self.V/2.0;                                                       # Normalize the covariance matrices
        V_2 = rho_2.V/2.0;
        
        OMEGA = self.Omega;
        OMEGA_T = np.transpose(OMEGA)
        
        delta_u = u_2 - u_1;                                                    # A bunch of auxiliar variables
        delta_u_T = np.hstack(delta_u)
        
        inv_V = np.linalg.inv(V_1 + V_2);
        
        V_aux = np.matmul( np.matmul(OMEGA_T, inv_V), OMEGA/4 + np.matmul(np.matmul(V_2, OMEGA), V_1) )
        
        identity = np.identity(2*self.N_modes);
        
        # V_temp = np.linalg.pinv(np.matmul(V_aux,OMEGA))                         # Trying to bypass singular matrix inversion ! I probably shouldnt do this...
        # F_tot_4 = np.linalg.det( 2*np.matmul(sqrtm(identity + matrix_power(V_temp                ,+2)/4) + identity, V_aux) );
        F_tot_4 = np.linalg.det( 2*np.matmul(sqrtm(identity + matrix_power(np.matmul(V_aux,OMEGA),-2)/4) + identity, V_aux) );
        
        F_0 = (F_tot_4.real / np.linalg.det(V_1+V_2))**(1.0/4.0);               # We take only the real part of F_tot_4 as there can be a residual complex part from numerical calculations!
        
        F = F_0*np.exp( -np.matmul(np.matmul(delta_u_T,inv_V), delta_u)  / 4);                        # Fidelity
        return F
    
    # Gaussian unitaries (applicable to single mode states)
    def displace(self, alpha, modes=[0]):
        """
        Apply displacement operator
       
        ARGUMENT:
           alpha - complex amplitudes for the displacement operator
           modes - indexes for the modes to be displaced 
        """
        
        if not (isinstance(alpha, list) or isinstance(alpha, np.ndarray) or isinstance(alpha, range)):      # Make sure the input variables are of the correct type
            alpha = [alpha]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == len(alpha), "Unable to decide which modes to displace nor by how much" # If the size of the inputs are different, there is no way of telling exactly what it is expected to do
        
        for i in range(len(alpha)):                                             # For each displacement amplitude
            idx = modes[i]                                                      # Get its corresponding mode
            
            d = 2.0*np.array([[alpha[i].real], [alpha[i].imag]]);               # Discover by how much this mode is to be displaced
            self.R[2*idx:2*idx+2] = self.R[2*idx:2*idx+2] + d;                  # Displace its mean value (covariance matrix is not altered)
    
    def squeeze(self, r, modes=[0]):
        """
        Apply squeezing operator on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           r     - ampllitude for the squeezing operator
           modes - indexes for the modes to be squeezed
        """
        
        if not (isinstance(r, list) or isinstance(r, np.ndarray) or isinstance(r, range)):              # Make sure the input variables are of the correct type
            r = [r]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == len(r), "Unable to decide which modes to squeeze nor by how much" # If the size of the inputs are different, there is no way of telling exactly what it is expected to do
        
        S = np.eye(2*self.N_modes)                                              # Build the squeezing matrix (initially a identity matrix because there is no squeezing to be applied on other modes)
        for i in range(len(r)):                                                 # For each squeezing parameter
            idx = modes[i]                                                      # Get its corresponding mode
            
            S[2*idx:2*idx+2, 2*idx:2*idx+2] = np.diag([np.exp(-r[i]), np.exp(+r[i])]); # Build the submatrix that squeezes the desired modes
        
        self.R = np.matmul(S, self.R);                                          # Apply squeezing operator on first  moments
        self.V = np.matmul( np.matmul(S,self.V), S);                            # Apply squeezing operator on second moments
        
    def rotate(self, theta, modes=[0]):
        """
        Apply phase rotation on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           theta - ampllitude for the rotation operator
           modes - indexes for the modes to be squeezed
        """
        
        if not (isinstance(theta, list) or isinstance(theta, np.ndarray) or isinstance(theta, range)):      # Make sure the input variables are of the correct type
            theta = [theta]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == len(theta), "Unable to decide which modes to rotate nor by how much" # If the size of the inputs are different, there is no way of telling exactly what it is expected to do
        
        Rot = np.eye(2*self.N_modes)                                            # Build the rotation matrix (initially identity matrix because there is no rotation to be applied on other modes)
        for i in range(len(theta)):                                             # For each rotation angle
            idx = modes[i]                                                      # Get its corresponding mode
            
            Rot[2*idx:2*idx+2, 2*idx:2*idx+2] = np.array([[np.cos(theta[i]), np.sin(theta[i])], [-np.sin(theta[i]), np.cos(theta[i])]]); # Build the submatrix that rotates the desired modes
        
        Rot_T = np.transpose(Rot)
        
        self.R = np.matmul(Rot, self.R);                                        # Apply rotation operator on first  moments
        self.V = np.matmul( np.matmul(Rot, self.V), Rot_T);                     # Apply rotation operator on second moments
        
    def phase(self, theta, modes=[0]):
        """
        Apply phase rotation on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           theta - ampllitude for the rotation operator
           modes - indexes for the modes to be squeezed
        """
        self.rotate(theta, modes)                                               # They are the same method/operator, this is essentially just a alias
    
    # Gaussian unitaries (applicable to two mode states)
    def beam_splitter(self, tau, modes=[0, 1]):
        """
        Apply a beam splitter transformation to pair of modes in a multimode gaussian state
        
        ARGUMENT:
           tau   - transmissivity of the beam splitter
           modes - indexes for the pair of modes which will receive the beam splitter operator 
        """
        
        # if not (isinstance(tau, list) or isinstance(tau, np.ndarray)):          # Make sure the input variables are of the correct type
        #     tau = [tau]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == 2, "Unable to decide which modes to apply beam splitter operator nor by how much"
        
        BS = np.eye(2*self.N_modes)
        i = modes[0]
        j = modes[1] 
        
        # B = np.sqrt(tau)*np.identity(2)
        # S = np.sqrt(1-tau)*np.identity(2)
        
        # BS[2*i:2*i+2, 2*i:2*i+2] = B
        # BS[2*j:2*j+2, 2*j:2*j+2] = B
        
        # BS[2*i:2*i+2, 2*j:2*j+2] =  S
        # BS[2*j:2*j+2, 2*i:2*i+2] = -S
        
        ##########################################
        sin_theta = np.sqrt(tau)
        cos_theta = np.sqrt(1-tau)
        
        BS[2*i  , 2*i  ] = sin_theta
        BS[2*i+1, 2*i+1] = sin_theta
        BS[2*j  , 2*j  ] = sin_theta
        BS[2*j+1, 2*j+1] = sin_theta
        
        BS[2*i+1, 2*j  ] = +cos_theta
        BS[2*j+1, 2*i  ] = +cos_theta
        
        BS[2*i  , 2*j+1] = -cos_theta
        BS[2*j  , 2*i+1] = -cos_theta
        ##########################################
        
        BS_T = np.transpose(BS)
        
        self.R = np.matmul(BS, self.R);
        self.V = np.matmul( np.matmul(BS, self.V), BS_T);
    
    def two_mode_squeezing(self, r, modes=[0, 1]):
        """
        Apply a two mode squeezing operator  in a gaussian state
        r - squeezing parameter
        
        ARGUMENT:
           r - ampllitude for the two-mode squeezing operator
        """
        
        # if not (isinstance(r, list) or isinstance(r, np.ndarray)):              # Make sure the input variables are of the correct type
        #     r = [r]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == 2, "Unable to decide which modes to apply two-mode squeezing operator nor by how much"
        
        S2 = np.eye(2*self.N_modes)
        i = modes[0]
        j = modes[1] 
        
        S0 = np.cosh(r)*np.identity(2);
        S1 = np.sinh(r)*np.diag([+1,-1]);
        
        S2[2*i:2*i+2, 2*i:2*i+2] = S0
        S2[2*j:2*j+2, 2*j:2*j+2] = S0
        
        S2[2*i:2*i+2, 2*j:2*j+2] = S1
        S2[2*j:2*j+2, 2*i:2*i+2] = S1
        
        # S2 = np.block([[S0, S1], [S1, S0]])
        S2_T = np.transpose(S2)
        
        self.R = np.matmul(S2, self.R);
        self.V = np.matmul( np.matmul(S2, self.V), S2_T)
        
    # Generic multimode gaussian unitary
    def apply_unitary(self, S, d):
        """
        Apply a generic gaussian unitary on the gaussian state
        
        ARGUMENTS:
            S,d - affine symplectic map (S, d) acting on the phase space, equivalent to gaussian unitary
        """
        assert all(np.isreal(d)) , "Error when applying generic unitary, displacement d is not real!"
        
        S_is_symplectic = np.allclose(np.matmul(np.matmul(S, self.Omega), S.transpose()), self.Omega)
        
        assert S_is_symplectic , "Error when applying generic unitary, unitary S is not symplectic!"
        
        self.R = np.matmul(S, self.R) + d
        self.V = np.matmul(np.matmul(S, self.V), S.transpose())
        
    # Gaussian measurements
    def measurement_general(self, *args):
        """
        After a general gaussian measurement is performed on the last m modes of a (n+m)-mode gaussian state
        this method calculates the conditional state the remaining n modes evolve into
        
        The user must provide the gaussian_state of the measured m-mode state or its mean value and covariance matrix
        
        At the moment, this method can only perform the measurement on the last modes of the global state,
        if you know how to perform this task on a generic mode, contact me so I can implement it! :)
       
        ARGUMENTS:
           R_m      - first moments     of the conditional state after the measurement
           V_m      - covariance matrix of the conditional state after the measurement
           or
           rho_m    - conditional gaussian state after the measurement on the last m modes (rho_B.N_modes = m)
        
        REFERENCE:
           Jinglei Zhang's PhD Thesis - https://phys.au.dk/fileadmin/user_upload/Phd_thesis/thesis.pdf
           Conditional and unconditional Gaussian quantum dynamics - Contemp. Phys. 57, 331 (2016)
        """
        if isinstance(args[0], gaussian_state):                                 # If the input argument is a gaussian_state
            R_m   = args[0].R;
            V_m   = args[0].V;
            rho_m = args[0]
        else:                                                                   # If the input arguments are the conditional state's mean quadrature vector anc covariance matrix
            R_m = args[0];
            V_m = args[1];
            rho_m = gaussian_state(R_m, V_m)
        
        idx_modes = range(int(self.N_modes-len(R_m)/2), self.N_modes);          # Indexes to the modes that are to be measured
        
        rho_B = only_modes(self, idx_modes);                                    # Get the mode measured mode in the global state previous to the measurement
        rho_A = partial_trace(self, idx_modes);                                 # Get the other modes in the global state        previous to the measurement
        
        n = 2*rho_A.N_modes;                                                    # Twice the number of modes in state A
        m = 2*rho_B.N_modes;                                                    # Twice the number of modes in state B
        
        V_AB = self.V[0:n, n:(n+m)];                                            # Get the matrix dictating the correlations      previous to the measurement                           
        
        inv_aux = np.linalg.inv(rho_B.V + V_m)                                  # Auxiliar variable
        
        # Update the other modes conditioned on the measurement results
        rho_A.R = rho_A.R - np.matmul(V_AB, np.linalg.solve(rho_B.V + V_m, rho_B.R - R_m) );
        
        rho_A.V = rho_A.V - np.matmul(V_AB, np.matmul(inv_aux, V_AB.transpose()) );
        
        rho_A.tensor_product([rho_m])                                           # Generate the post measurement gaussian state
        
        self.R = rho_A.R                                                        # Copy its attributes into the original instance
        self.V = rho_A.V
        self.Omega   = rho_A.Omega
        self.N_modes = rho_A.N_modes
    
    def measurement_homodyne(self, *args):
        """
        After a homodyne measurement is performed on the last m modes of a (n+m)-mode gaussian state
        this method calculates the conditional state the remaining n modes evolve into
        
        The user must provide the gaussian_state of the measured m-mode state or its mean quadrature vector
        
        At the moment, this method can only perform the measurement on the last modes of the global state,
        if you know how to perform this task on a generic mode, contact me so I can implement it! :)
       
        ARGUMENTS:
           R_m      - first moments of the conditional state after the measurement (assumes measurement on position quadrature
           or
           rho_m    - conditional gaussian state after the measurement on the last m modes (rho_B.N_modes = m)
        
        REFERENCE:
           Jinglei Zhang's PhD Thesis - https://phys.au.dk/fileadmin/user_upload/Phd_thesis/thesis.pdf
        """
      
        if isinstance(args[0], gaussian_state):                                 # If the input argument is a gaussian_state
            R_m   = args[0].R;
            rho_m = args[0]
        else:                                                                   # If the input argument is the mean quadrature vector
            R_m = args[0];
            V_m = args[1];
            rho_m = gaussian_state(R_m, V_m)
        
        idx_modes = range(int(self.N_modes-len(R_m)/2), self.N_modes);          # Indexes to the modes that are to be measured
        
        rho_B = only_modes(self, idx_modes);                                    # Get the mode measured mode in the global state previous to the measurement
        rho_A = partial_trace(self, idx_modes);                                 # Get the other modes in the global state        previous to the measurement
        
        n = 2*rho_A.N_modes;                                                    # Twice the number of modes in state A
        m = 2*rho_B.N_modes;                                                    # Twice the number of modes in state B
        
        V_AB = self.V[0:n, n:(n+m)];                                            # Get the matrix dictating the correlations      previous to the measurement
        
        MP_inverse = np.diag([1/rho_B.V[1,1], 0]);                              # Moore-Penrose pseudo-inverse an auxiliar matrix (see reference)
        
        rho_A.R = rho_A.R - np.matmul(V_AB, np.matmul(MP_inverse, rho_B.R - R_m   ) ); # Update the other modes conditioned on the measurement results
        rho_A.V = rho_A.V - np.matmul(V_AB, np.matmul(MP_inverse, V_AB.transpose()) );
        
        rho_A.tensor_product([rho_m])                                           # Generate the post measurement gaussian state
        
        self.R = rho_A.R                                                        # Copy its attributes into the original instance
        self.V = rho_A.V
        self.Omega   = rho_A.Omega
        self.N_modes = rho_A.N_modes
    
    def measurement_heterodyne(self, *args):
        """
        After a heterodyne measurement is performed on the last m modes of a (n+m)-mode gaussian state
        this method calculates the conditional state the remaining n modes evolve into
        
        The user must provide the gaussian_state of the measured m-mode state or the measured complex amplitude of the resulting coherent state
        
        At the moment, this method can only perform the measurement on the last modes of the global state,
        if you know how to perform this task on a generic mode, contact me so I can implement it! :)
       
        ARGUMENTS:
           alpha    - complex amplitude of the coherent state after the measurement
           or
           rho_m    - conditional gaussian state after the measurement on the last m modes (rho_m.N_modes = m)
        
        REFERENCE:
           Jinglei Zhang's PhD Thesis - https://phys.au.dk/fileadmin/user_upload/Phd_thesis/thesis.pdf
        """
        
        if isinstance(args[0], gaussian_state):                                 # If the input argument is  a gaussian_state
            rho_m = args[0];
        else:
            rho_m = gaussian_state("coherent", args[0]);
        
        self.measurement_general(rho_m);
        
    
    # Phase space representation
    def wigner(self, X, P):
        """
        Calculates the wigner function for a single mode gaussian state
       
        PARAMETERS
            X, P - 2D grid where the wigner function is to be evaluated (use meshgrid)
        
        CALCULATES:
            W - array with Wigner function over the input 2D grid
        """
        
        assert self.N_modes == 1, "At the moment, this program only calculates the wigner function for a single mode state"
        
        N = self.N_modes;                                                       # Number of modes
        W = np.zeros((len(X), len(P)));                                         # Variable to store the calculated wigner function
        
        one_over_purity = 1/self.purity();
        
        inv_V = np.linalg.inv(self.V)
        
        for i in range(len(X)):
            x = np.block([ [X[i,:]] , [P[i,:]] ]);   
            
            for j in range(x.shape[1]):
                dx = np.vstack(x[:, j]) - self.R;                                          # x_mean(:,i) is the i-th point in phase space
                dx_T = np.hstack(dx)
                
                W_num = np.exp( - np.matmul(np.matmul(dx_T, inv_V), dx)/2 );    # Numerator
                
                W_den = (2*np.pi)**N * one_over_purity;                         # Denominator
          
                W[i, j] = W_num/W_den;                                          # Calculate the wigner function at every point on the grid
        return W
    
    def q_function(self, *args):
        """
        Calculates the Hussimi Q-function over a meshgrid
        
        PARAMETERS (numpy.ndarray, preferably generated by np.meshgrid):
            X, Y - 2D real grid where the Q-function is to be evaluated (use meshgrid to generate the values on the axes)
            OR
            ALPHA - 2D comples grid, each entry on this matrix is a vertex on the grid (equivalent to ALPHA = X + 1j*Y)
        
        CALCULATES:
            q_func - array with q-function over the input 2D grid
           
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
            Many thanks to Daniel Tandeitnik for the base code for this method!
        """
        
        # Handle input into correct form
        if len(args) > 1:                                                      # If user passed more than one argument (should be X and Y - real values of on the real and imaginary axes)
            X = args[0]
            Y = args[1]
            ALPHA = X + 1j*Y                                                    # Then, construct the complex grid
        else:
            ALPHA = args[0]                                                     # If the user passed a single argument, it should be the complex grid, just rename it
        
        ALPHA = np.array(ALPHA)                                                 # Make sure ALPHA is the correct type
        
        # Preamble, get auxiliar variables that depend only on the gaussian state parameters
        one_over_sqrt_2 = 1.0/np.sqrt(2)                                        # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = one_over_sqrt_2*np.block([[-1j*eye_N, +1j*eye_N],
                                      [    eye_N,     eye_N]]);                 # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
        
        Q = np.zeros([2*self.N_modes,1])                                        # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable
        
        # Loop through the meshgrid and evaluate Q-function
        q_func = np.zeros(ALPHA.shape)
        
        for i in range(ALPHA.shape[0]):
            for j in range(ALPHA.shape[1]):
        
                gamma = np.zeros(2*self.N_modes,dtype=np.complex_)              # Auxiliar 2*self.N_modes complex vector      
                gamma[:self.N_modes] = np.conj(ALPHA[i, j])                     # First N entries are the complex conjugate of alpha
                gamma[self.N_modes:] = ALPHA[i, j]                              # Last  N entries are alpha
                
                q_func[i,j] = np.real(P_0*np.exp( -0.5*np.matmul(np.conj(gamma),gamma) -0.5*np.matmul( gamma , np.matmul(R,gamma)) + np.matmul( gamma , np.matmul(R,y)) ))
        
        q_func = q_func / (np.pi**self.N_modes)
        
        return q_func
    
    # Density matrix elements
    def density_matrix_coherent_basis(self, alpha, beta):
        """
        Calculates the matrix elements of the density operator on the coherent state basis
        
        PARAMETERS:
            alpha - a N-array with complex aplitudes (1xN numpy.ndarray)
            beta - a N-array with complex aplitudes (NxN numpy.ndarray)
        
        CALCULATES:
            q_f - the matrix element \bra{\alpha}\rho\kat{\beta}
           
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
            Many thanks to Daniel Tandeitnik for the base code for this method!
        """
        
        assert (len(alpha) == len(beta)) and (len(alpha) == self.N_modes), "Wrong input dimensions for the matrix element of the density matrix in coherent state basis!"
        
        one_over_sqrt_2 = 1.0/np.sqrt(2)                                        # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = np.block([[-1j*one_over_sqrt_2*eye_N, +1j*one_over_sqrt_2*eye_N],
                      [    one_over_sqrt_2*eye_N,     one_over_sqrt_2*eye_N]]); # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
        
        Q = np.zeros(2*self.N_modes)                                            # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable
        
        gamma = np.zeros(2*self.N_modes,dtype=np.complex_)                      # Auxiliar 2*self.N_modes complex vector      
        gamma[:self.N_modes] = np.conj(beta)                                    # First N entries are the complex conjugate of beta
        gamma[self.N_modes:] = alpha                                            # Last  N entries are alpha
        
        beta_rho_alpha = P_0*np.exp( -0.5*np.matmul(np.conj(gamma),gamma) -0.5*np.matmul( gamma , np.matmul(R,gamma)) + np.matmul( gamma , np.matmul(R,y)) ) # Hussimi Q-function
        
        return beta_rho_alpha
    
    def density_matrix_number_basis(self, n_cutoff=10):
        """
        Calculates the number distribution of the gaussian state
        
        PARAMETERS:
            n_cutoff - maximum number for the calculation
            
        RETURNS:
            P - array with the number distribution of the state (P.shape = self.N_modes*[n_cutoff])
            
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
        """
        
        # Preamble, get auxiliar variables that depend only on the gaussian state parameters
        one_over_sqrt_2 = 1.0/np.sqrt(2)                                        # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = np.block([[-1j*one_over_sqrt_2*eye_N, +1j*one_over_sqrt_2*eye_N],
                      [    one_over_sqrt_2*eye_N,     one_over_sqrt_2*eye_N]]); # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
        
        Q = np.zeros([2*self.N_modes,1])                                        # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable        
        
        H = Hermite_multidimensional(R, y, n_cutoff)                            # Calculate the Hermite polynomial associated with this gaussian state
        
        # Calculate the probabilities
        rho_m_n = np.zeros((2*self.N_modes)*[n_cutoff])                         # Initialize the tensor to 0 (n_cutoff entries in each of the 2*self.N_modes dimensions)
        
        # rho is the same shape as H !
        
        m_last = np.array((2*self.N_modes)*[0], dtype=int)
        idx = np.ravel_multi_index(list(m_last), dims=rho_m_n.shape, order='F') # Get "linearized" index
        rho_m_n.ravel()[idx] = P_0                                              # Set its first entry to P_0

        # Similar procedure to what precedes. Move forward in the P tensor and fill it element by element.
        # next_m = np.ones([self.N_modes, 1], dtype=int);
        # next_n = np.ones([self.N_modes, 1], dtype=int);
        
        n_entries = np.prod(H.shape)                                            # Number of entries on the multidimensional Hermite polynomial tensor
        
        for mn_linear in range(0, n_entries):                                   # Loop through every entry on tensor H using linear indices ( m is the linearized index for H: H.ravel()[m] <-> H[ np.unravel_index(m, H.shape, order='F') ] )
        
            mn = np.array(np.unravel_index(mn_linear, H.shape, order='F'), dtype=int)  # Vector index for the next entry of the Hermite tensor to be calculated
            
            m = mn[:self.N_modes]                                               # First self.N_modes entries are the vector m
            n = mn[self.N_modes:]                                               # Last  self.N_modes entries are the vector n
            
            big_factorial = 1.0
            for kk in range(self.N_modes):                                      # Next, divide by the square root of the appropriate factorial! # kk = 1:dim/2
                big_factorial = big_factorial*np.math.factorial(m[kk])*np.math.factorial(n[kk]);
            
            rho_m_n.ravel()[mn_linear] = P_0*H.ravel()[mn_linear]/np.sqrt(big_factorial)
                    
        return rho_m_n 
    
    def number_statistics(self, n_cutoff=10):
        """
        Calculates the number distribution of the gaussian state
        
        PARAMETERS:
            n_cutoff - maximum number for the calculation
            
        RETURNS:
            P - array with the number distribution of the state (P.shape = self.N_modes*[n_cutoff])
            
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
        """
        
        # Preamble, get auxiliar variables that depend only on the gaussian state parameters
        one_over_sqrt_2 = 1.0/np.sqrt(2.0)                                      # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = one_over_sqrt_2*np.block([[-1j*eye_N, +1j*eye_N],
                                      [    eye_N,     eye_N]]);                 # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
            
        Q = np.zeros([2*self.N_modes,1])                                        # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable
        
        # DEBUGGING !
        # R_old = np.matmul( np.matmul( np.conj(np.transpose(U)) , (1+1e-15)*eye_2N-2*M) , np.matmul( np.linalg.pinv(eye_2N+2*M) , np.conj(U) ) ) # Auxiliar variable
        # y_old = 2*np.matmul( np.matmul( np.transpose(U) , np.linalg.pinv((1+1e-15)*eye_2N-2*M) ) , Q )                                          # Auxiliar variable
        # P_0_old = ( (det(M + 0.5*eye_2N))**(-0.5) )*np.exp( -1*np.matmul( Q.transpose() , np.matmul( np.linalg.pinv(2*M + eye_2N) , Q )  ) )                # Auxiliar variable
        # 
        # assert np.allclose(R  ,   R_old, rtol=1e-10, atol=1e-10), "Achei!"
        # assert np.allclose(y  ,   y_old, rtol=1e-10, atol=1e-10), "Achei!"
        # assert np.allclose(P_0, P_0_old, rtol=1e-10, atol=1e-10), "Achei!"
        # 
        # print("Passou")
        
        H = Hermite_multidimensional(R, y, n_cutoff)                            # Calculate the Hermite polynomial associated with this gaussian state
        
        # Calculate the probabilities
        P = np.zeros(self.N_modes*[n_cutoff])                                   # Initialize the tensor to 0 (n_cutoff entries in each of the self.N_modes dimensions)
        
        idx = np.ravel_multi_index((self.N_modes)*[0], dims=P.shape, order='F')            # Get "linearized" index
        P.ravel()[idx] = P_0                                                    # Set its first entry to P_0

        # Similar procedure to what precedes. Move forward in the P tensor and fill it element by element.
        nextP = np.ones([self.N_modes, 1], dtype=int);
        for jj in range(1, 1+n_cutoff**(self.N_modes)-1):            # jj = 1:n_cutoff^(dim/2) - 1
            
            
            for ii in range(1, 1+self.N_modes):                      #ii = 1:dim/2   # Figure out what the next coordinate to fill in is
                jumpTo = np.zeros([self.N_modes, 1], dtype=int);
                jumpTo[ii-1] = 1;
                
                if nextP[ii-1] + jumpTo[ii-1] > n_cutoff:
                   nextP[ii-1] = 1;
                else:
                   nextP[ii-1] = nextP[ii-1] + 1;
                   break
            
            nextCoord = np.ravel_multi_index(list(nextP-1), dims=P.shape, order='F')    # Get "linearized" index
            
            whichH = np.zeros([2*self.N_modes, 1], dtype=int);                  # Corresponding entry on Hermite polynomial
            whichH[:self.N_modes] = nextP                                    # Copy the position of the probability twice                 
            whichH[self.N_modes:] = nextP                                    # whichH = [nextP, nextP]
            
            # whichH = np.zeros([2*self.N_modes, 1], dtype=int);
            # for kk in range(self.N_modes):              # m = (n,n) -> Repeat the entries 
            #     whichH[kk] = nextP[kk];                 # m[0:N]   = n
            #     whichH[kk+self.N_modes] = nextP[kk];    # m[N:2*N] = n
            
            idx_H = np.ravel_multi_index(list(whichH-1), dims=H.shape, order='F') # Get "linearized" index whichH = num2cell(whichH);
            
            P.ravel()[nextCoord] = P_0*H.ravel()[idx_H];
            for kk in range(self.N_modes):                     # kk = 1:dim/2
                P.ravel()[int(nextCoord)] = P.ravel()[int(nextCoord)]/np.math.factorial(int(nextP[kk]-1));
        
        return P

