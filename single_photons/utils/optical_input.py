import numpy as np
from .constants import c


def compute_optical_input(alpha_in, kappa, laser_linewidth, delta_t, N, detuning):
    theta = np.arctan(detuning/kappa)
    optical_noise = np.sqrt(laser_linewidth)*alpha_in*np.random.normal()
    x_in = (
        np.sqrt(kappa) * delta_t * (
            (1/np.sqrt(kappa*delta_t))*np.sin(theta)*optical_noise
            + np.conjugate(alpha_in)
            + alpha_in
            )
    )
    y_in = (
        1j
        * np.sqrt(kappa)
        * delta_t
        * (
            (1/np.sqrt(kappa*delta_t))*np.cos(theta)*optical_noise
            + np.conjugate(alpha_in)
            - alpha_in
        )
    )
    optical_input = 1j*np.zeros((4, N))
    optical_input[0, :] = x_in
    optical_input[1, :] = y_in
    return optical_input

def create_pulse(photon_number, cavity_linewidth, laser_linewidth,
                 t, cavity_length, detuning):
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
