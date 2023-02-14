# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 19:53:53 2021

@author: calif
"""

import math
import numpy as np
import quantum_gaussian_toolbox as qgt
import matplotlib.pyplot as plt


# parameters and auxiliary variables

# mechanical mode freqs
omega1 = 1;
omega2 = 0.9;

# optical loss rate
kappa = 3*omega1;

# feedback gain
g_cd1 = 0.8;
g_cd2 = 0.8;

G1 = 0.16*omega1; # effective optomechanical coupling
G2 = 0.10*omega1;

gamma1 = 0.00004*omega1; # mechanical mode damping coefficients
gamma2 = 0.00003*omega1;

Gamma11 = gamma1 + g_cd1*G1*omega1/kappa; # effective damping matrix elements
Gamma22 = gamma2 + g_cd2*G2*omega2/kappa;
Gamma12 = g_cd1*G2*omega2/kappa;
Gamma21 = g_cd2*G1*omega1/kappa;

c = 3.93*10**13; # k_b*T/hbar in Hz
n1 = c/omega1; # occupation numbers for initial thermal states
n2 = c/omega2;
D11 = gamma1*(2*n1+1) + G1**2/kappa;
D12 = G1*G2/kappa;
D22 = gamma2*(2*n2+1) + G2**2/kappa;
step = 0.05/omega1; #time step
nsteps = 7001;
t = np.linspace(0, (nsteps-1)*step, nsteps);
nbar = {};
n_mean = {};
n_variance = {};
lower_fill1 = np.zeros(len(t));
lower_fill2 = np.zeros(len(t));

# boolean for logarithmic scale (better visualization)
logscale = bool(1)

# dynamic operators
A = np.array([[0,        omega1,     0,       0],
              [-omega1, -Gamma11,    0,    -Gamma12], 
              [0,          0,        0,      omega2],
              [0,       -Gamma21, -omega2, -Gamma22]])
N = np.zeros((4,1));
D = np.array([[0, 0, 0, 0],
             [0, D11, 0, D12],
             [0, 0, 0, 0],
             [0, D12, 0, D22]]);

# initial state
particle1 = qgt.gaussian_state("thermal", n1) # mode 1 initial state
particle2 = qgt.gaussian_state("thermal", n2) # mode 2 initial state
initial = qgt.tensor_product([particle1, particle2]) # total initial state

#simulation
simulation = qgt.gaussian_dynamics(A, D, N, initial)
states = simulation.unconditional_dynamics(t)

#nbar = states.occupation_number()
for i in range(len(t)):
    nbar[i] = states[i].occupation_number()
    n_mean[i], n_variance[i] = states[i].number_operator_moments()
    lower_fill1[i] = float(nbar[i][0]) - np.sqrt(float(np.real(n_variance[i][0][0])))
    lower_fill2[i] = float(nbar[i][1]) - np.sqrt(float(np.real(n_variance[i][1][1])))

idx = np.argwhere(lower_fill1<=0)
lower_fill1[idx] = 1

idx = np.argwhere(lower_fill2<=0)
lower_fill2[idx] = 1

ss = simulation.steady_state()
n_mec_final = ss.occupation_number()

fig1, ax1 = plt.subplots()
data_n = np.zeros((2,len(t))) #stores occupation number data
var_1 = np.zeros((2, len(t))) #stores variance data for mode 1
var_2 = np.zeros((2, len(t))) #stores variance data for mode 2
for i in range(len(t)):
    if logscale:
        data_n[0,i] = math.log(nbar[i][0]);
        data_n[1,i] = math.log(nbar[i][1]);
        var_1[0,i] = math.log(nbar[i][0] + math.sqrt(np.real(n_variance[i][0][0])));
        var_1[1,i] = math.log(lower_fill1[i]);
        var_2[0,i] = math.log(nbar[i][1] + math.sqrt(np.real(n_variance[i][1][1])));
        var_2[1,i] = math.log(lower_fill2[i]);
    else:
        data_n[0,i] = nbar[i][0];
        data_n[1,i] = nbar[i][1]
        var_1[0,i] = nbar[i][0] + math.sqrt(np.real(n_variance[i][0][0]));
        var_1[1,i] = lower_fill1[i];
        var_2[0,i] = nbar[i][1] + math.sqrt(np.real(n_variance[i][1][1]));
        var_2[1,i] = lower_fill2[i];

plt.figure()
plt.xlim([0, 350]);
if logscale:
    plt.ylim([23, 32]);
    plt.ylabel('ln(occupation number)')
else:
    plt.xlim([0, 200]);
    plt.ylabel('occupation number')
plt.xlabel('time steps')
plt.plot(t[:], data_n[0,:], color = 'darkred')
plt.fill_between(t[:], var_1[0,:], var_1[1,:], color = 'red', alpha=0.15);
plt.plot(t[:], data_n[1,:], color = 'blue')
plt.fill_between(t[:], var_2[0,:], var_2[1,:], color = 'blue', alpha = 0.15);
plt.hlines(y=math.log(n_mec_final[0]), xmin=0, xmax=(nsteps-1)*step, colors='red', linestyles='-', lw=2)
plt.hlines(y=math.log(n_mec_final[1]), xmin=0, xmax=(nsteps-1)*step, colors='deepskyblue', linestyles='-', lw=2)
plt.tight_layout()