import numpy as np
from numpy.linalg import det
from numpy.linalg import matrix_power
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power
from .gaussian_state import gaussian_state
from single_photons.utils.is_a_function import is_a_function
from .lyapunov import lyapunov_ode_conditional, lyapunov_ode_unconditional
from single_photons.states import *

class gaussian_dynamics:
    """Class simulating unconditional and conditional dynamics of a gaussian state following a set of Langevin and Lyapunov equations
    
    ATTRIBUTES
        A                     - Drift matrix (can be a callable lambda functions to have a time dependency!)
        D                     - Diffusion Matrix 
        N                     - Mean values of the noises
        initial_state         - Initial state of the global system
        t                     - Array with timestamps for the time evolution
        
        is_stable             - Boolean telling if the system is stable or not
        N_time                - Length of time array
        Size_matrices         - Size of covariance, diffusion and drift matrices
        
        states_unconditional  - List of time evolved states following unconditional dynamics
        states_conditional    - List of time evolved states following   conditional dynamics (mean quadratures are the average of the trajectories from the quantum monte carlo method)
        steady_state_internal - Steady state
        
        quantum_trajectories        - Quantum trajectories from the Monte Carlo method for the conditional dynamics
        semi_classical_trajectories - List of time evolved semi-classical mean quadratures (Semi-classical Monte Carlo method)
    """
    
    def __init__(self, A_0, D_0, N_0, initial_state_0):
        """Class constructor for simulating the time evolution of the multimode systems following open unconditional and conditional quantum dynamics dictated by Langevin and Lyapunov equations
        
        Langevin: \dot{R} = A*X + N           : time evolution of the mean quadratures
       
        Lyapunov: \dot{V} = A*V + V*A^T + D   : time evolution of the covariance matrix
       
        PARAMETERS:
           A_0           - Drift matrix     (numerical matrix or callable function for a time-dependent matrix)
           D_0           - Diffusion Matrix (auto correlation of the noises, assumed to be delta-correlated in time)
           N_0           - Mean values of the noises
           initial_state - Cavity linewidth
       
        BUILDS:
           self           - instance of a time_evolution class
           self.is_stable - boolean telling if the system is stable or not
        """
      
        self.A = A_0;  # .copy() ?                                              # Drift matrix
        self.D = D_0;  # .copy() ?                                              # Diffusion Matrix
        self.N = N_0.reshape((len(N_0),1));   # .copy() ?                       # Mean values of the noises
        
        self.initial_state = initial_state_0;                                   # Initial state of the global system
        
        self.Size_matrices = len(self.D);                                       # Size of system and ccupation number for the environment (heat bath)
      
        # assert 2*initial_state_0.N_modes == self.Size_matrices), "Initial state's number of modes does not match the drift and diffusion matrices sizes"              # Check if the initial state and diffusion/drift matrices have appropriate sizes !
      
        if( not is_a_function(self.A) ):
            eigvalue, eigvector = np.linalg.eig(self.A);                        # Calculate the eigenvalues of the drift matrix
            is_not_stable = np.any( eigvalue.real > 0 );                        # Check if any eigenvalue has positive real part (unstability)
            self.is_stable = not is_not_stable                                  # Store the information of the stability of the system in a class attribute
    
    def unconditional_dynamics(self, t_span):
        """Calculates the time evolution of the initial state following an unconditional dynamics at the input timestamps.
        
       PARAMETERS:
           tspan - Array with time stamps when the calculations should be done
       
       CALCULATES: 
           self.states_conditional - list of gaussian_state instances with the time evolved gaussian states for each timestamp of the input argument t_span
       
        RETURNS:
            result - list of gaussian_state instances with the time evolved gaussian states for each timestamp of the input argument t_span
        """
      
        R_evolved, status_langevin = self.langevin(t_span);                     # Calculate the mean quadratures for each timestamp
      
        V_evolved, status_lyapunov = self.lyapunov(t_span);                     # Calculate the CM for each timestamp (perform time integration of the Lyapunov equation)
        
        assert status_langevin != -1 and status_lyapunov != -1, "Unable to perform the time evolution of the unconditional dynamics - Integration step failed"         # Make sure the parameters for the time evolution are on the correct order of magnitude!
                
        self.states_unconditional = []                                          # Combine the time evolutions calculated above into an array of gaussian states
        for i in range(self.N_time):
            self.states_unconditional.append( gaussian_state(R_evolved[:, i], V_evolved[i]) );
        
        result = self.states_unconditional;
        return result                                                           # Return the array of time evolved gaussian_state following unconditional dynamics
    
    def langevin(self, t_span):
        """Solve quantum Langevin equations for the time evolved mean quadratures of the multimode systems
       
        Uses ode45 to numerically integrate the average Langevin equations (a fourth order Runge-Kutta method)
       
        PARAMETERS:
            t_span - timestamps when the time evolution is to be calculated (ndarray)
       
        CALCULATES:
            self.R - a cell with the time evolved mean quadratures where self.R(i,j) is the i-th mean quadrature at the j-th timestamp
        """
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
            langevin_ode = lambda t, R: np.reshape(np.matmul(self.A(t), R.reshape((len(R),1))) + self.N, (len(R),))        # Function handle that defines the Langevin equation (returns the derivative)
        else:
            langevin_ode = lambda t, R: np.reshape(np.matmul(self.A, np.reshape(R, (len(R),1))) + self.N, (len(R),))           # Function handle that defines the Langevin equation (returns the derivative)
        
        solution_langevin = solve_ivp(langevin_ode, [t_span[0], t_span[-1]], np.reshape(self.initial_state.R, (self.Size_matrices,)), t_eval=t_span) # Solve Langevin eqaution through Runge Kutta(4,5)
        # Each row in R corresponds to the solution at the value returned in the corresponding row of self.t
        
        R_evolved = solution_langevin.y;                                        # Store the time evolved quadratures in a class attribute
        
        return R_evolved, solution_langevin.status
    
    def lyapunov(self, t_span, is_conditional=False, AA=0, DD=0, B=0):
        """Solve the lyapunov equation for the time evolved covariance matrix of the full system (both conditional and unconditional cases)
       
        Uses ode45 to numerically integrate the Lyapunov equation, a fourth order Runge-Kutta method
       
        PARAMETERS:
            t_span - timestamps when the time evolution is to be calculated
       
        CALCULATES:
            'self.V' - a cell with the time evolved covariance matrix where self.V[j] is the covariance matrix at the j-th timestamp in t_span
        """
        
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        V_0_vector = np.reshape(self.initial_state.V, (self.Size_matrices**2, )); # Reshape the initial condition into a vector (expected input for ode45)
        
        if is_conditional:                                                      # Check if the dynamics is conditional or unconditional
            if is_a_function(self.A):                                           # Check if there is a time_dependency on the odes
                ode = lambda t, V: lyapunov_ode_conditional(t, V, self.A(t)+AA, self.D+DD, B); # Function handle that defines the Langevin equation (returns the derivative)
            else:
                ode = lambda t, V: lyapunov_ode_conditional(t, V, self.A+AA   , self.D+DD, B);    # Lambda unction that defines the Lyapunov equation (returns the derivative)
        else:
            if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
                ode = lambda t, V: lyapunov_ode_unconditional(t, V, self.A(t), self.D); # Function handle that defines the Langevin equation (returns the derivative)
            else:
                ode = lambda t, V: lyapunov_ode_unconditional(t, V, self.A, self.D);    # Lambda unction that defines the Lyapunov equation (returns the derivative)
                
        solution_lyapunov = solve_ivp(ode, [t_span[0], t_span[-1]], V_0_vector, t_eval=t_span) # Solve Lyapunov equation through Fourth order Runge Kutta
        
        # Unpack the output of ode45 into a list where each entry contains the information about the evolved CM at each time
        V_evolved = []                                                          # Initialize a cell to store the time evolvd CMs for each time
        
        for i in range(len(solution_lyapunov.t)):
            V_current_vector = solution_lyapunov.y[:,i];                                        # Take the full Covariance matrix in vector form
            V_current = np.reshape(V_current_vector, (self.Size_matrices, self.Size_matrices)); # Reshape it into a proper matrix
            V_evolved.append(V_current);                                                        # Append it on the class attribute
                    
        return V_evolved, solution_lyapunov.status
        
    def steady_state(self, A_0=0, A_c=0, A_s=0, omega=0): # *args -> CONSERTAR !
        """Calculates the steady state for the multimode system
       
        PARAMETERS (only needed if the drift matrix has a periodic time dependency):
          A_0, A_c, A_s - Components of the Floquet decomposition of the drift matrix
          omega         - Frequency of the drift matrix
        
        CALCULATES:
            self.steady_state_internal - gaussian_state with steady state of the system
          
        RETURNS:
            ss - gaussian_state with steady state of the system
        """
      
        if is_a_function(self.A):                                               # If the Langevin and Lyapunov eqs. have a time dependency, move to the Floquet solution
            ss = self.floquet(A_0, A_c, A_s, omega);
            self.steady_state_internal = ss;                                    # Store it in the class instance
        
        else:                                                                   # If the above odes are time independent, 
            assert self.is_stable, "There is no steady state covariance matrix, as the system is not stable!"  # Check if there exist a steady state!
        
            R_ss = np.linalg.solve(self.A, -self.N);                            # Calculate steady-state mean quadratures
            V_ss = solve_continuous_lyapunov(self.A, -self.D);                  # Calculate steady-state covariance matrix
        
            ss = gaussian_state(R_ss, V_ss);                                    # Generate the steady state
            self.steady_state_internal = ss;                                    # Store it in the class instance
            
        return ss                                                               # Return the gaussian_state with the steady state for this system
    
    def floquet(self, A_0, A_c, A_s, omega):
        """Calculates the staeady state of a system with periodic Hamiltonin/drift matrix
        
        Uses first order approximation in Floquet space for this calculation
       
        Higher order approximations will be implemented in the future
        
        PARAMETERS:
          A_0, A_c, A_s - components of the Floquet decomposition of the drift matrix
          omega - Frequency of the drift matrix
        
        CALCULATES:
          self.steady_state_internal - gaussian_state with steady state of the system
          
        RETURNS:
          ss - gaussian_state with steady state of the system
        """
      
        M = self.Size_matrices;                                                 # Size of the time-dependent matrix
        Id = np.identity(M);                                                    # Identity matrix for the system size
        
        A_F = np.block([[A_0,    A_c   ,     A_s  ],
                        [A_c,    A_0   , -omega*Id],
                        [A_s, +omega*Id,     A_0  ]])                           # Floquet drift     matrix
        
        D_F = np.kron(np.eye(3,dtype=int), self.D)                              # Floquet diffusion matrix
        
        N_F = np.vstack([self.N, self.N, self.N])                               # Floquet mean noise vector
        
        R_ss_F = np.linalg.solve(A_F, -N_F);                                    # Calculate steady-state Floquet mean quadratures vector
        V_ss_F = solve_continuous_lyapunov(A_F, -D_F);                          # Calculate steady-state Floquet covariance matrix
        
        R_ss = R_ss_F[0:M];                                                     # Get only the first entries
        V_ss = V_ss_F[0:M, 0:M];                                                # Get only the first sub-matrix
        
        ss = gaussian_state(R_ss, V_ss);                                        # Generate the steady state
        self.steady_state_internal = ss;                                        # Store it in the class instance
        
        return ss
    
    def semi_classical(self, t_span, N_ensemble=2e+2):
        """Solve the semi-classical Langevin equation for the expectation value of the quadrature operators using a Monte Carlos simulation to numerically integrate the Langevin equations
        
        The initial conditions follows the initial state probability density in phase space
        The differential stochastic equations are solved through a Euler-Maruyama method
       
        PARAMETERS:
          t_span - timestamps when the time evolution is to be calculated
          N_ensemble (optional) - number of iterations for Monte Carlos simulation, default value: 200
       
        CALCULATES:
          self.R_semi_classical - matrix with the quadratures expectation values of the time evolved system where 
          self.R_semi_classical(i,j) is the i-th quadrature expectation value at the j-th time
        """
      
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        dt = self.t[2] - self.t[1];                                             # Time step
        sq_dt =  np.sqrt(dt);                                                   # Square root of time step (for Wiener proccess in the stochastic integration)
        
        noise_amplitude = self.N + np.sqrt( np.diag(self.D) );                  # Amplitude for the noises (square root of the auto correlations)
        
        mean_0 = self.initial_state.R;                                          # Initial mean value
        std_deviation_0 =  np.sqrt( np.diag(self.initial_state.V) );            # Initial standard deviation
        
        self.semi_classical_trajectories = np.zeros((self.Size_matrices, self.N_time));    # Matrix to store each quadrature ensemble average at each time
        
        if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
            AA = lambda t: self.A(t);                                           # Rename the function that calculates the drift matrix at each time
        else:
            AA = lambda t: self.A;                                              # If A is does not vary in time, the new function always returns the same value 
      
        for i in range(N_ensemble):                                             # Loop on the random initial positions (# Monte Carlos simulation using Euler-Maruyama method in each iteration)
            
            X = np.zeros((self.Size_matrices, self.N_time));                    # For this iteration, this matrix stores each quadrature at each time (first and second dimensions, respectively)
            X[:,0] = np.random.normal(mean_0, std_deviation_0)                  # Initial Cavity position quadrature (normal distribution for vacuum state)
            
            noise = np.random.standard_normal(X.shape);
            for k in range(self.N_time-1):                                      # Euler-Maruyama method for stochastic integration
                X[:,k+1] = X[:,k] + (np.matmul(AA(self.t[k]), X[:,k]) + self.N)*dt + sq_dt*np.multiply(noise_amplitude, noise[:,k])
                                   
            self.semi_classical_trajectories = self.semi_classical_trajectories + X;    # Add the new  Monte Carlos iteration quadratures to the same matrix
        
        self.semi_classical_trajectories = self.semi_classical_trajectories/N_ensemble; # Divide the ensemble sum to obtain the average quadratures at each time
        
        result = self.semi_classical_trajectories
        return result
        
    def langevin_conditional(self, t_span, V_evolved, N_ensemble=200, rho_bath=gaussian_state(), C=0, Gamma=0, V_m=0):
        """Solve the conditional stochastic Langevin equation for the expectation value of the quadrature operators
        using a Monte Carlos simulation to numericaly integrate the stochastic Langevin equations
        
        The differential stochastic equations are solved through a Euler-Maruyama method
       
        PARAMETERS:
          t_span     - timestamps when the time evolution is to be calculated
          N_ensemble - number of iterations for Monte Carlos simulation, default value: 200
          rho_bath   - gaussian_state with the quantum state of the environment's state
          C          - matrix describing the measurement process (see conditional_dynamics)
          Gamma      - matrix describing the measurement process (see conditional_dynamics)
          V_m        - Covariance matrix of the post measurement state
       
        CALCULATES:
          self.quantum_trajectories - list of single realizations of the quantum Monte Carlo method for the mean quadrature vector
        
        RETURN:
          R_conditional - average over the trajectories of the quadrature expectation values
        """
        
        N_ensemble = int(N_ensemble)
        
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        dt = self.t[2] - self.t[1];                                             # Time step
        sq_dt_2 =  np.sqrt(dt)/2.0;                                             # Square root of time step (for Wiener proccess in the stochastic integration)
        
        R_conditional = np.zeros((self.Size_matrices, self.N_time));            # Matrix to store each quadrature ensemble average at each time
        
        if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
            AA = lambda t: self.A(t);                                           # Rename the function that calculates the drift matrix at each time
        else:
            AA = lambda t: self.A;                                              # If A is does not vary in time, the new function always returns the same value 
        
        N_measured2 = C.shape[0]                                                 # Number of modes to be measured
        C_T = np.transpose(C)
        Gamma_T = np.transpose(Gamma)
        
        R_bath = np.reshape(rho_bath.R, (len(rho_bath.R),))                                   # Mean quadratures for the bath state
        V_bath = rho_bath.V
        
        N = np.reshape(self.N, (len(self.N),))
        
        self.quantum_trajectories = N_ensemble*[None]                           # Preallocate memory to store the trajectories
        
        for i in range(N_ensemble):                                             # Loop on the random initial positions (# Monte Carlos simulation using Euler-Maruyama method in each iteration)
            
            X = np.zeros((self.Size_matrices, self.N_time));                    # Current quatum trajectory to be calculated, this matrix stores each quadrature at each time (first and second dimensions, respectively)
            X[:,0] = np.reshape(self.initial_state.R, (2*self.initial_state.N_modes,))                     # Initial mean quadratures are exactly the same as the initial state (stochasticity only appear on the measurement outcomes)
            
            cov = (V_bath + V_m)/2.0                                            # Covariance for the distribution of the measurement outcome (R_m)
            R_m = np.random.multivariate_normal(R_bath, cov, (self.N_time))     # Sort the measurement results
            
            for k in range(self.N_time-1):                                      # Euler-Maruyama method for stochastic integration of the conditional Langevin equation
                V = V_evolved[k]                                       # Get current covariance matrix (pre-calculated according to deterministic conditional Lyapunov equation)
                dw = np.matmul(fractional_matrix_power(V_bath+V_m, -0.5), R_m[k,:] - R_bath) # Calculate the Wiener increment
                
                X[:,k+1] = X[:,k] + (np.matmul(AA(self.t[k]), X[:,k]) + N)*dt + sq_dt_2*np.matmul(np.matmul(V, C_T) + Gamma_T, dw) # Calculate the quantum trajectories following the stochastis conditional Langevin equation
            
            self.quantum_trajectories[i] = X                                    # Store each trajectory into class instance                
            
            R_conditional = R_conditional + X;                                  # Add the new quantum trajectory in order to calculate the average
        
        R_conditional = R_conditional/N_ensemble;                               # Divide the ensemble sum to obtain the average quadratures at each time
        
        return R_conditional
    
    def conditional_dynamics(self, t_span, N_ensemble=1e+2, C_int=None, rho_bath=None, s_list = [1], phi_list=None):
        """Calculates the time evolution of the initial state following a conditional dynamics at the input timestamps
        
        Independent general-dyne measurements can be applied to each mode of the multimode gassian state with N modes
        
        PARAMETERS:
            tspan    - numpy.ndarray with time stamps when the calculations should be done
            C_int    - interaction matrix between system and bath
            rho_bath - gaussian state of the bath
            s        - list of measurement parameters for each measured mode (s=1: Heterodyne ; s=0: Homodyne in x-quadrature ; s=Inf: Homodyne in p-quadrature)
            phi      - list of directions on phase space of the measurement for each measured mode
            
        CALCULATES:
            self.states_conditional - list of time evolved gaussian_state for each timestamp of the input argument t_span
            
        RETURNS:
            result - list of time evolved gaussian_state for each timestamp of the input argument t_span
        """
        
        # TODO: generalize for arbitrary number of monitored modes        
        # TODO: Independent measurements can be applied to the last k modes of the multimode gassian state with N modes
        # N_measured = len(s_list)                                              # Number of monitored modes
        # Omega_m = rho_bath.Omega                                              # Symplectic form matrix for the monitored modes
        # Omega_n = self.initial_state.Omega                                    # Symplectic form matrix for the whole system
        # C = np.matmul(np.matmul(temp, Omega_m), C_int_T);                     # Extra matrix on the Lyapunov equation
        # Gamma = -np.matmul(np.matmul(np.matmul(temp, V_bath),C_int_T),Omega_n)# Extra matrix on the Lyapunov equation
        print(self.initial_state.N_modes)
        N_measured = self.initial_state.N_modes                                 # Number of monitored modes (currently all modes)
        Omega = self.initial_state.Omega                                        # Symplectic form matrix for the whole system
        
        if phi_list is None: phi_list = N_measured*[0]                          # If no measurement direction was indicated, use default value of 0 to all measured modes
        
        if C_int is None: C_int = np.eye(2*N_measured)                          # If no system-bath interaction bath was indicated, use as default value an identity matrix
        
        if rho_bath is None: rho_bath = vacuum(N_measured)                      # If no state for the environment was indicated, use  as default value a tensor product of vacuum states
        
        assert N_measured == len(phi_list), "conditional_dynamics can not infer the number of modes to be measured, number of measurement parameters is different from rotation angles"
        print(rho_bath.N_modes)

        assert rho_bath.N_modes == N_measured, "The number of bath modes does not match the number of monitored modes"
        assert N_measured <= self.initial_state.N_modes, "There are more monitored modes, than there are on the initial state"
        
        V_bath = rho_bath.V                                                     # Covariance matrix for the bath's state
        
        V_m = 42                                                                # Just to initialize the variable, this value will be removed after the loop
        for i in range(N_measured):                                             # For each measurement parameter
            s = s_list[i]
            temp = np.block([[s, 0],[0, 1/s]])                                  # Calculate the associated mode's covariance matrix after measurement
            
            phi = phi_list[i]                                                   # Get rotation of measurement angle
            Rot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
            Rot_T = np.transpose(Rot)
            
            temp = np.matmul( np.matmul(Rot, temp), Rot_T)                      # Rotate measured covariance matrix
            
            V_m = block_diag(temp, V_m)                                         # Build the covariance matrix of the tensor product of these modes
        V_m = V_m[0:len(V_m)-1, 0:len(V_m)-1]                                   # Remove the initialization value
        
        temp = np.linalg.inv(V_bath + V_m)                     # Auxiliar variable
        temp_minus = fractional_matrix_power(V_bath + V_m, -0.5)
        
        B = np.matmul(np.matmul(C_int, Omega), temp_minus)                        # Extra matrix on the Lyapunov equation
        
        AA = - np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(Omega, C_int), V_bath), temp), Omega), np.transpose(C_int))
        DD = + np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(Omega, C_int), V_bath), temp), V_bath), np.transpose(C_int)), Omega)
        
        is_conditional = True                                                   # Boolean telling auxiliar variable that the conditional dynamics is to be calculated
        
        V_evolved, status_lyapunov = self.lyapunov(t_span, is_conditional, AA, DD, B);       # Calculate the deterministic dynamics for the CM for each timestamp (perform time integration of the Lyapunov equation)
        
        assert status_lyapunov != -1, "Unable to perform the time evolution of the covariance matrix through Lyapunov equation - Integration step failed"
        
        ################################################################################################################################
        C = np.transpose(-np.matmul(np.matmul(C_int, Omega), temp))              # Extra matrix on the Lyapunov equation
        
        Gamma = np.transpose(np.matmul(np.matmul(np.matmul(Omega, C_int), V_bath), temp)) # Extra matrix on the Lyapunov equation
        ################################################################################################################################
        
        R_evolved = self.langevin_conditional(t_span, V_evolved, N_ensemble, rho_bath, C, Gamma, V_m)  # Calculate the quantum trajectories and its average
        
        self.states_conditional = []                                            # Combine the time evolutions calculated above into an array of gaussian states
        for i in range(self.N_time):
            self.states_conditional.append( gaussian_state(R_evolved[:, i], V_evolved[i]) );        
      
        result = self.states_conditional;
        return result                                                           # Return the array of time evolved gaussian_state