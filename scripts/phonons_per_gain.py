import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from control import dare
from single_photons.estimators.kalman import KalmanFilter
import single_photons.utils.constants as ct
from single_photons.utils.parameters import *
from single_photons.environment import Particle


def discrete_integration(env, Ad, Bd, Q, R, G, t, std_detection, N):
    x0 = 0
    P0 = 3e8*np.matrix(np.eye(2))
    estimation = np.matrix([[x0*np.random.normal()], [x0*np.random.normal()]])
    states = np.array([[x0*np.random.normal()], [x0*np.random.normal()]])
    new_states = np.zeros((N, 2))
    measured_states = np.zeros((N))
    estimated_states = np.zeros((N, 2))
    estimated_states[0, :] = estimation.reshape((2))
    estimation = estimation.reshape((2, 1))
    control = np.array([[0]])
    controls = []
    kalman = KalmanFilter(estimation, P0, Ad, Bd, env.C, Q, R)
    for i in tqdm(range(t.shape[0])):
        new_states[i, :] = states[:, 0]
        if not i % control_step:
            measured_states[i] = states[0, 0] + std_detection * np.random.normal()
            kalman.propagate_dynamics(control)
            kalman.compute_aposteriori(measured_states[i])
            estimated_states[i, :] = kalman.estimates_aposteriori[int(i/control_step)][:, 0].reshape((2))
            estimation = estimated_states[i, :].reshape((2, 1))
            control = -np.matmul(G, estimation)
        else:
            measured_states[i] = measured_states[i-1]
            estimated_states[i, :] = estimated_states[i-1,:]
        controls.append(float(control))
        states = env.step(states, control=control, delta_t=delta_t)
    return new_states, measured_states, estimated_states, kalman, controls


def run_simulation(power, wavelength, waist, radius, pressure,
                   fs, eta_detection, control_step, delta_t):
    gamma, omega, ba_force, std_detection, std_z = compute_parameters_simulation(power, wavelength, waist, 
                                                                      radius, pressure, fs, eta_detection)  
    period = 2*np.pi/omega
    t = np.arange(0, 20 * period, delta_t)
    N = t.shape[0]
    coupling = (1/(4*np.pi))*(ba_force**2)
    env = Particle(omega, gamma, coupling, radius=radius, eta_detection=1, T=293)
    variance_process = env.thermal_force_std**2 + env.backaction_std**2
    std_detection = std_detection/env.zp_x
    Q = np.array([[0, 0], [0, variance_process]])*control_step*delta_t/2
    R = np.array([[np.power(std_detection,2)]])
    g_fb = 2*omega
    Ad = scipy.linalg.expm(env.A *control_step*delta_t)
    Bd = env.B * delta_t * control_step
    cost_states = np.array([[omega/2, 0],
                            [0, omega/2]])
    X, L, G = dare(Ad, Bd, cost_states, omega/(g_fb**2))
    sim_states, measure_states, est_states, kalman, control = discrete_integration(env, Ad, Bd, 
                                                                                   Q, R, G, t, 
                                                                                   std_detection, N)
    

if __name__=='__main__':
    p = 0
    radius = 75e-9
    wavelength = 1.064e-6
    power = 300e-3
    waist = 0.6e-6
    eta_detection = 1
    delta_t = 1e-9
    control_step = 30 
    fs = 1/(control_step*delta_t)