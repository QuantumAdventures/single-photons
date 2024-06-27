import numpy as np
import scipy
from single_photons.utils.parameters import *
from .constants import c

def compute_optical_input(N, delta_t, env, laser_linewidth, photon_number, cavity_length, wavelength, duration):
    theta = np.arctan(env.__detuning__/env.__kappa__)
    
    Ad_sim = scipy.linalg.expm(env.A * delta_t)
    optical_injection_matrix = (Ad_sim - np.eye(4)) @ np.linalg.pinv(env.A)
    FSR, f, r, I_factor = compute_cavity_parameters(env.__kappa__, cavity_length, env.__detuning__, wavelength)
    amplitude = np.sqrt(photon_number * I_factor)
    optical_input = 1j*np.zeros((4,N))
    end = int(duration/delta_t)
    end = min(N,int(N/2) + end)
    s = end - int(N/2)
    noise = np.random.normal(size = s)
    optical_input[0,int(N/2):end] = np.array(s*[(np.conjugate(amplitude) + amplitude) * np.sqrt(env.__kappa__)]) + np.sin(theta)*np.sqrt(laser_linewidth)*amplitude*noise
    optical_input[1,int(N/2):end] = np.array(s*[1j*(np.conjugate(amplitude) - amplitude) * np.sqrt(env.__kappa__)]) + np.cos(theta)*np.sqrt(laser_linewidth)*amplitude*noise
    for i in range(int(N/2), end):
        x,y = optical_input[0:2,i]
        optical_input[0:2, i] = optical_injection_matrix[:2,:2] @ np.array([x,y])
    return FSR, f, r, I_factor, optical_input

def create_pulse(photon_number, cavity_linewidth, laser_linewidth,
                 t, cavity_length, detuning):
    amplitude = photon_number**2
    N = t.shape[0]
    delta_t = np.diff(t)[0]
    amplitude = photon_number**2
    center = int(N/2)
    round_trip_time = 2*cavity_length/c
    trips = 1/(2*np.pi*round_trip_time*cavity_linewidth)
    width = trips*round_trip_time/delta_t
    alpha_in = amplitude/(
        delta_t * np.power(2*np.pi*width**2,0.5)) *\
        np.exp(-np.power(np.arange(0, N, 1, dtype=np.int64) - center,2)/(2*width**2))
    optical_input = compute_optical_input(alpha_in, cavity_linewidth,
                                          laser_linewidth, delta_t, N, detuning)
    return optical_input, center, width
