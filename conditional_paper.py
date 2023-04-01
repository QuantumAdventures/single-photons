"""
Example of usage of the gaussian_dynamics class to calculate a conditional dynamics
Github: https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python
Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation 
# from matplotlib import cm

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np
from single_photons import gaussian_dynamics
from single_photons.utils import *
from single_photons.states import *
from single_photons.utils.operations import *    
from single_photons.utils.properties import *

gamma = 2*np.pi*10
nbar_env = 0
chi = gamma/3

A = np.block([[-chi-gamma/2, 0],
            [0, chi-gamma/2]])

D = gamma*(2*nbar_env+1)*np.eye(2)
N = np.zeros((2,1))
initial_state = coherent(alpha=3)
t = np.linspace(0, 0.36, 2000)
simulation = gaussian_dynamics(A, D, N, initial_state)
C = np.diag([np.sqrt(gamma), np.sqrt(gamma)])
rho_b = thermal(nbar_env)
conditional_states = simulation.conditional_dynamics(t, N_ensemble=100, C_int=C, rho_bath=rho_b, s_list=[1e-5], phi_list=[np.pi/2])

conditional_sq = np.zeros(len(t))

for i in range(N_time):
    pass