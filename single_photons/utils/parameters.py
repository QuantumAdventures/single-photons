import numpy as np
from .constants import *

def compute_gamma(radius, pressure, rho=2200, T = 293, index_refraction=1.444):
    m_p  = rho * 4 * np.pi * np.power(radius, 3) / 3
    v_gas = np.sqrt(3 * kb * T / m_gas)
    gamma = 15.8 * radius**2 * pressure / (m_p * v_gas)
    return gamma

def compute_omega(power, waist, rho=2200, index_refraction=1.444):
    omega = (
    np.sqrt(12 / np.pi)
    * np.sqrt((index_refraction**2 - 1) / (index_refraction**2 + 2)) ** 3
    * np.sqrt(power)
    / (waist**2 * np.sqrt(rho * c))
    )
    return omega

def compute_phonons(estimations, cov_matrix, step=100):
    sampled_cov_matrix = cov_matrix[::step]
    N = len(sampled_cov_matrix)
    phonons = np.zeros((N-1))
    for i in range(1, N):
        averaged = estimations[(i-1)*step:i*step, :].mean(axis=0)
        second_moments = sampled_cov_matrix[i]+np.power(averaged,2)
        phonons[i-1] = np.trace(second_moments)/4-0.5
    return phonons

def compute_scattered_power(power, waist, wavelength, rho=2200, index_refraction=1.444):
    Nm = rho/(amu*60.08) #SiO2 molecular density
    k_tweezer = 2*np.pi/wavelength
    pol_permit_ratio = 3/Nm*(index_refraction**-1)/(index_refraction**2+2) #from C-M
    sigma = (8*np.pi/3)*(pol_permit_ratio*k_tweezer*k_tweezer/(4*np.pi*8.85e-12))**2    
    I0 = 2*power/(np.pi*waist)
    return 0.1*I0*sigma

def compute_backaction(wavelength, p_scat, A=0.71):
    k = 2*np.pi/wavelength
    ba_force = np.sqrt(2*(A**2 + 0.4)*hbar*k*p_scat/c)
    return ba_force

def compute_ideal_detection(wavelength, p_scat, A=0.71):
    k = 2*np.pi/wavelength
    return np.sqrt(2*hbar*c/((A**2+0.4)*4*k*p_scat))

def compute_parameters_simulation(power, wavelength, tweezer_waist, radius, pressure, fs, eta_detection,
                                  rho=2200, index_refraction=1.444, T=293):
    gamma = compute_gamma(radius, pressure, rho=rho, T=T, index_refraction=index_refraction)
    omega = compute_omega(power, tweezer_waist, rho=rho, index_refraction=index_refraction)
    p_scat = compute_scattered_power(power, tweezer_waist, wavelength, rho=rho, index_refraction=index_refraction)
    ba_force = compute_backaction(wavelength, p_scat)
    std_z = compute_ideal_detection(wavelength, p_scat)
    std_detection = std_z*np.sqrt(fs/(2*eta_detection))
    return gamma, omega, ba_force, std_detection, std_z

def compute_parameters_simulation_cavity(power, wavelength, tweezer_waist, radius, pressure, fs, eta_detection,\
                                  cavity_length, cavity_waist, detuning_ratio, cavity_linewidth_ratio,\
                                  rho=2200, index_refraction=1.444, T=293):
    gamma = compute_gamma(radius, pressure, rho=rho, T=T, index_refraction=index_refraction)
    omega = compute_omega(power, tweezer_waist, rho=rho, index_refraction=index_refraction)
    p_scat = compute_scattered_power(power, tweezer_waist, wavelength, rho=rho, index_refraction=index_refraction)
    ba_force = compute_backaction(wavelength, p_scat)
    std_z = compute_ideal_detection(wavelength, p_scat)
    std_detection = std_z*np.sqrt(fs/(2*eta_detection))
    detuning = omega*detuning_ratio
    cavity_linewidth = omega*cavity_linewidth_ratio
    cavity_freq = detuning + 2*np.pi*c/wavelength
    g_cs = (
    np.power(12 / np.pi, 1 / 4)
    * np.power((index_refraction**2 - 1) / (index_refraction**2 + 2), 3 / 4)
    * np.power(power * radius**6 * cavity_freq**6 / (c**5 * rho), 1 / 4)
    / (np.sqrt(cavity_length) * cavity_waist))
    return gamma, omega, ba_force, std_detection, std_z, g_cs, detuning, cavity_linewidth
