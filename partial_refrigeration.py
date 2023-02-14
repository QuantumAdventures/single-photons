# -*- coding: utf-8 -*-
"""
Example of usage of the gaussian_dynamics class
Github: https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python

Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib import cm
from matplotlib.patches import Circle

import numpy as np
import quantum_gaussian_toolbox as qgt


##### Parameters
omega = 2*np.pi                                 # Particle natural frequency 
gamma = 2*np.pi*0.3                             # Mechanical damping constant 
kappa = 2*np.pi*0.1                             # Optical damping constant
G     = 2*np.pi
n_th  = 10

alpha = 2                                       # Initial complx amplitude
n_0   = 10                                      # Particle initial occupation number

t = np.linspace(0, 7*np.pi/omega, int(2000));   # Timestamps for simulation


###### Matrix definning the dynamics
A = np.array([[    0     ,   omega     ,      0   ,     0  ],
              [  -omega  ,   -gamma    ,      G   ,     0  ],
              [    0     ,     0       ,   -kappa ,     0  ], 
              [    G     ,     0       ,      0   , -kappa ]]) # Drift matrix for harmonic potential with damping

N = np.zeros((4,1));                # Driving vector

D = np.diag([0, gamma*(2*n_th+1), 2*kappa, 2*kappa]);        # Diffusion matrix


##### Simulation
cavity   = qgt.squeezed(r=1.2)          # Initial cavity state
particle = qgt.thermal(n_0)             # Initial particle state

initial  = qgt.tensor_product([particle, cavity]) # Initial state
                               
simulation = qgt.gaussian_dynamics(A, D, N, initial) # Create instance of time evolution of gaussian state
states = simulation.unconditional_dynamics(t)        # Simulate

n_mec = np.zeros(len(t))            # List to store occupation numbers
n_opt = np.zeros(len(t))

for i in range(len(t)):             # Loopt through time-evolved states and calculate their occupation numbers
    [n_mec[i], n_opt[i]] = states[i].occupation_number()
    
    
ss = simulation.steady_state()
n_mec_final = ss.occupation_number()[0]



############################### Plotting ###############################


##### Plot_expectation_values(result_ref);
plt.figure()
plt.plot(t, n_mec, lw=3.0, color = 'C0', label = 'QuGIT')
plt.hlines(n_mec_final, t[0], t[-1])
plt.hlines(n_th, t[0], t[-1])

ax = plt.gca()
# ax.fill_between(t, n_bar - np.sqrt(n_var), n_bar + np.sqrt(n_var), alpha=0.15);

plt.xlabel(r'$\omega_{0} t$')
plt.ylabel(r'$\langle n \rangle $')
plt.legend(loc='upper right', fontsize=14)
plt.grid(True, which="both", ls="-", alpha = 0.2)

plt.title('Number operator moments')

plt.savefig("occupation_number_damped_unconditional_dynamics", format='svg')





##### Time-dependent matrix
# A = np.array([[-kappa/2,  Delta , 0, 0,0,0], [ -Delta , -kappa/2, -2*g1, 0, 0, 0 ],[ 0 ,  0, 0, omega1, 0, 0 ],[ -2*g1 ,  0, -omega1, -gamma1, 0,0 ],[ 0 ,  0,0, 0, 0, omega2 ],[ 0 ,  0, 0, 0, -omega2, -gamma2 ]]);                     


# def func_RWA(t, A, g2 , delta):
    

#     A[0][4] = -2*g2*np.sin(delta*t);
#     A[1][4] = -2*g2*np.cos(delta*t);

#     A[5][0] =  -2*g2*np.cos(delta*t);
#     A[5][1]  =  +2*g2*np.sin(delta*t);
    
#     return A


# A_rwa = lambda t : func_RWA(t, A, g2, delta);


##### Plot fidelity
# plt.figure()
# plt.plot(t, F_vac, lw=3.0, color = 'C0') #, label = 'QuGIT')

# plt.xlabel(r'$\omega_{0} t$')
# plt.ylabel(r'Fidelity with vacuum')
# #plt.legend(loc='upper right', fontsize=14)
# plt.grid(True, which="both", ls="-", alpha = 0.2)


# ##### Plot Wigner functions
# x = np.linspace(-6,6,200)
# y = np.linspace(-6,6,200)
# X, P = np.meshgrid(x, y);


# n=int(len(t)/8)
# fig, axs = plt.subplots(nrows=1, ncols=int(len(t)/n-1))

# for i in range(0, len(t), n):
#         W = states[i].wigner(X, P);
        
#         idx = int(i/n)
        
#         if idx == 7:
#             break
        
#         axs[idx].imshow(W, cmap='plasma', interpolation='nearest', origin='lower',extent=[min(x),max(x),min(x),max(x)])
#         circ = Circle((0, 0), 2*np.abs(alpha), facecolor='None', edgecolor='k', linestyle='--', lw=2, alpha = 0.4,)
#         axs[idx].add_patch(circ)
#         if idx > 0:
#             axs[idx].get_yaxis().set_ticks([])
#             # axs[idx].set_axis_off()

# plt.tight_layout(pad=10.0)

# plt.savefig("wigner_damped_unconditional_dynamics", format='svg')

############################### Plotting ###############################



