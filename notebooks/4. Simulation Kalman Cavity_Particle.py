#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from single_photons.estimators.kalman import KalmanFilter
import single_photons.utils.constants as ct
from single_photons.environment import Cavity_Particle


# In[2]:

T = 293.15
m_gas = ct.amu*(0.01*40 + 0.78*28 + 0.21*32)
v_gas = np.sqrt(3*ct.kb*T/m_gas)
p = 1e2
R = 147e-9
rho = 2200
m_p = 4/3*np.pi*R**3*rho
tweezer_wavelength = 1.55e-6
tweezer_freq = 2*np.pi*ct.c/tweezer_wavelength
index_refraction = 1.4440
tweezer_power = 200e-3
tweezer_waist = 0.6e-6
cavity_waist = 100e-6
cavity_length = 50e-3
coupling = 0.01
eta_detec = 0.9

gamma = 15.8*R**2*p/(m_p*v_gas)
omega = np.sqrt(12/np.pi)*np.sqrt((index_refraction-1)/(index_refraction+2))**3*\
    np.sqrt(tweezer_power)/(tweezer_waist**2*np.sqrt(rho*ct.c))
detuning = 0.3*omega
cavity_linewidth = omega

cavity_freq = detuning + tweezer_freq
g_cs = np.power(12/np.pi,1/4)*np.power((index_refraction-1)/(index_refraction+2),3/4)*\
    np.power(tweezer_power*R**6*cavity_freq**6/(ct.c**5*rho),1/4)/(np.sqrt(cavity_length)*\
                                                                cavity_waist)        

g_fb = 1e5
period = 2*np.pi/omega
t = np.arange(0, 10*period, period/1000)
N = t.shape[0]
delta_t = np.diff(t)[0]


# In[56]:

env = Cavity_Particle(omega, gamma, detuning, cavity_linewidth, g_cs, coupling,\
                      radius = R, rho = rho, eta_detection = eta_detec)

pulse_amplitude = 1e3
pulse_center = 2500
pulse_width = 200
alpha_in = []
for i in range(t.shape[0]):
    #alpha = i*(t.shape[0]-i)*pulse_amplitude
    alpha = pulse_amplitude*(np.exp(-(i-pulse_center)**2/(2*pulse_width**2)))
                               #+np.exp(-(i-(pulse_center+t.shape[0]/2))**2/(2*pulse_width**2)))
    alpha_in.append(alpha)

# In[57]:


variance_process = 2*env.__gamma__ + np.power(env.backaction, 2)
std_detection = 10


# In[58]:


Q = np.array([[cavity_linewidth, 0, 0, 0],
              [0, cavity_linewidth, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, variance_process]])
R = np.array([[np.power(std_detection,2)]])


# In[59]:

Ad = scipy.linalg.expm(env.A*delta_t)

# In[60]:


P0 = 10*np.matrix(np.eye(4))
estimation = np.matrix([[0],[0],[0],[0]])
states = np.array([[0],[0],[100],[0.]])
K = np.array([[0, 0, 1, 1e5]])
new_states = np.zeros((N,4)) 
measured_states = np.zeros((N))
estimated_states = np.zeros((N, 4))
estimated_states[0,:] = estimation.reshape((4))
estimation = estimation.reshape((4,1))
control = np.array([[0]])
kalman = KalmanFilter(estimation, P0, Ad, env.B*delta_t, env.C, Q, R)
for i in tqdm(range(t.shape[0])):
    new_states[i,:] = states[:,0]
    measured_states[i] = states[2, 0] + std_detection*np.random.normal()
    kalman.propagate_dynamics(control)
    kalman.compute_aposteriori(measured_states[i])
    estimated_states[i,:] = kalman.estimates_aposteriori[i][:,0].reshape((4))
    estimation = estimated_states[i,:].reshape((4,1))    
    control = -g_fb*estimation[2]
    states = env.step(states, alpha_in = alpha_in[i], control = control, delta_t = delta_t)


# In[61]:
'''
plt.close('all')
fig1 = plt.Figure()
plt.title('Position')
plt.plot(t[:], measured_states[:])
plt.plot(t[:], estimated_states[:,2])
plt.plot(t[:], new_states[:,2])
plt.grid()
plt.legend(['Measured', 'Estimated','Simulated'])
plt.show()
'''

# In[62]:
'''
plt.figure()
#fig2 = plt.Figure()
plt.title('X quadrature')
plt.plot(t[:], estimated_states[:,0])
plt.plot(t[:], new_states[:,0])
plt.grid()
plt.legend(['Estimated','Simulated'])
plt.show()
'''

# In[63]:
'''
plt.figure()
plt.title('Photon number')
num_sim = []
num_est = []
for i in range(t.shape[0]):
    num_sim.append(estimated_states[i,0]**2 + estimated_states[i,1]**2)
    num_est.append(new_states[i,0]**2 + new_states[i,1]**2)
plt.plot(t[:], num_sim)
plt.plot(t[:], num_est)
plt.grid()
plt.legend(['Estimated','Simulated'])
plt.show()
'''

plt.title('Photon number')
num_sim = []
num_est = []
for i in range(t.shape[0]):
    if t[i] > 1:#2.1e-5:
        break
    num_sim.append(new_states[i,0]**2 + new_states[i,1]**2)
    #num_est.append(estimated_states[i,0]**2 + estimated_states[i,1]**2)
if i == t.shape[0]-1:
    i += 1
plt.plot(t[:i], num_sim)
#plt.plot(t[:], num_est)
plt.grid()
#plt.legend(['Estimated'])
plt.show()
