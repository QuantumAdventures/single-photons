#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from control import dare, lqr
import scipy
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from single_photons.estimators.kalman import KalmanFilter
import single_photons.utils.constants as ct
from single_photons.environment import Cavity_Particle


# In[2]:

T = 293
p = 1e-5
R = 147e-9
rho = 2200
index_refraction = 1.4440
tweezer_wavelength = 1.55e-6
tweezer_power = 200e-3
tweezer_waist = 0.6e-6
cavity_waist = 100e-6
cavity_length = 50e-3
eta_detec = 0.178

m_p = 4 / 3 * np.pi * R**3 * rho
v_gas = np.sqrt(3 * ct.kb * T / ct.m_gas)
Nm = rho / (ct.amu * 60.08)  # SiO2 molecular density
tweezer_freq = 2 * np.pi * ct.c / tweezer_wavelength
pol_permit_ratio = (
    3 / Nm * (index_refraction**2-1) / (index_refraction**2 + 2)
)  # from C-M
gamma = 15.8 * R**2 * p / (m_p * v_gas)
omega = (
    np.sqrt(12 / np.pi)
    * np.sqrt((index_refraction**2 - 1) / (index_refraction**2 + 2)) ** 3
    * np.sqrt(tweezer_power)
    / (tweezer_waist**2 * np.sqrt(rho * ct.c))
)
coupling = (
    9
    * pol_permit_ratio**2
    * tweezer_power
    * tweezer_freq**5
    / (128 * np.pi**2 * ct.c**6 * m_p * omega)
)
coupling = coupling/(ct.hbar/(2*m_p*omega))

detuning = 1 * omega
cavity_linewidth = omega
cavity_freq = detuning + tweezer_freq

g_cs = (
    np.power(12 / np.pi, 1 / 4)
    * np.power((index_refraction**2 - 1) / (index_refraction**2 + 2), 3 / 4)
    * np.power(tweezer_power * R**6 * cavity_freq**6 / (ct.c**5 * rho), 1 / 4)
    / (np.sqrt(cavity_length) * cavity_waist)
)
g_cs = 0

period = 2 * np.pi / omega
t = np.arange(0, 10 * period, period / 3000)
N = t.shape[0]
delta_t = np.diff(t)[0]


# In[56]-2

env = Cavity_Particle(
    omega,
    gamma,
    detuning,
    cavity_linewidth,
    g_cs,
    coupling,
    radius=R,
    rho=rho,
    eta_detection=eta_detec,
)

pulse_amplitude = 0
pulse_center = 2500
pulse_width = 200
alpha_in = []
for i in range(t.shape[0]):
    # alpha = i*(t.shape[0]-i)*pulse_amplitude
    alpha = pulse_amplitude * (
        np.exp(-((i - pulse_center) ** 2) / (2 * pulse_width**2))
    )
    # +np.exp(-(i-(pulse_center+t.shape[0]/2))**2/(2*pulse_width**2)))
    alpha_in.append(alpha)

# In[57]:

variance_process = 2 * env.__gamma__ + np.power(env.backaction, 2)
std_detection = 1

# In[58]:


Q = np.array(
    [
        [(env.__optical_noise__**2) * cavity_linewidth, 0, 0, 0],
        [0, (env.__optical_noise__**2) * cavity_linewidth, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, variance_process],
    ]
)
R = np.array([[np.power(std_detection, 2)]])


# In[59]:

Ad = scipy.linalg.expm(env.A * delta_t)
cost_states = np.array(
    [[0.00001, 0, 0, 0], [0, 0.00001, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
(G, S, E) = lqr(Ad, env.B * delta_t, cost_states, 1e5)

# In[60]:

x0 = 15
P0 = 1 * np.matrix(np.eye(4))
estimation = np.matrix([[0], [0], [x0], [0]])
states = np.array([[0], [0], [x0], [0.0]])
K = np.array([[0, 0, 1, 1e5]])
new_states = np.zeros((N, 4))
measured_states = np.zeros((N))
estimated_states = np.zeros((N, 4))
estimated_states[0, :] = estimation.reshape((4))
estimate_Vx = []
estimate_Vp = []
estimation = estimation.reshape((4, 1))
control = np.array([[0]])
controls = []
kalman = KalmanFilter(estimation, P0, Ad, env.B * delta_t, env.C, Q * delta_t, R)
for i in tqdm(range(t.shape[0])):
    new_states[i, :] = states[:, 0]
    measured_states[i] = states[2, 0] + std_detection * np.random.normal()
    kalman.propagate_dynamics(control)
    kalman.compute_aposteriori(measured_states[i])
    estimate_Vx.append(float(kalman.error_covariance_aposteriori[i][2,2]))
    estimate_Vp.append(float(kalman.error_covariance_aposteriori[i][3,3]))
    estimated_states[i, :] = kalman.estimates_aposteriori[i][:, 0].reshape((4))
    estimation = estimated_states[i, :].reshape((4, 1))
    control = -np.matmul(G, estimation)
    controls.append(float(control))
    states = env.step(states, alpha_in=alpha_in[i], control=control, delta_t=delta_t)


# In[61]:
plt.close("all")

plt.figure()
plt.title("Position")
plt.plot(t[1:], measured_states[1:], alpha=0.95)
plt.plot(t[1:], estimated_states[1:, 2], alpha=0.95)
plt.plot(t[1:], new_states[1:, 2], alpha=0.95)
C = max(controls) / 4
controls = [x / C for x in controls]
plt.plot(t[1:], controls[1:], alpha=0.5)
plt.grid()
plt.legend(["Measured", "Estimated", "Simulated", "Control input"])
plt.show()

# In[62]:

plt.figure()
plt.title("X quadrature")
plt.plot(t[1:], estimated_states[1:, 0])
plt.plot(t[1:], new_states[1:, 0])
plt.grid()
plt.legend(["Estimated", "Simulated"])
plt.show()


# In[63]:
    
plt.figure()
plt.title("Photon number")
plt.plot(
    t[1:], np.power(estimated_states[1:, 0], 2) + np.power(estimated_states[1:, 1], 2)
)
plt.plot(t[1:], np.power(new_states[1:, 0], 2) + np.power(new_states[1:, 1], 2))
plt.grid()
plt.legend(["Estimated", "Simulated"])
plt.show()


# In[64]:
fig4,ax = plt.subplots()
plt.title("Oscillator phase space")
plt.plot(estimated_states[1:,2],estimated_states[1:,3])
patches = []
for i in tqdm(range(t.shape[0])[1:]):
    radius = np.sqrt(estimate_Vx[i]) + np.sqrt(estimate_Vp[i])
    circle = matplotlib.patches.Circle((estimated_states[i,2], estimated_states[i,3])\
                                       ,radius)
    patches.append(circle)
coll = matplotlib.collections.PatchCollection(patches,facecolors = 'gray',alpha = 0.05)
ax.add_collection(coll)
plt.grid()
plt.show()
