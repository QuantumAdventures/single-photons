import numpy as np
from .constants import kb, hbar, c, amu, m_gas, epsilon_0

def compute_zpx(omega, radius, rho = 2200):
    m_p = rho * (4 / 3) * np.pi * np.power(radius,3)
    zp_x = np.sqrt(hbar / (2*m_p*omega))
    return zp_x

def compute_gamma(radius, pressure, rho=2200, T=293):
    m_p = rho * 4 * np.pi * np.power(radius, 3) / 3
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

def compute_scattered_power(
    power,
    waist,
    wavelength,
    rho=2200,
    index_refraction=1.444,
):
    Nm = rho / (amu * 60.08)
    k_tweezer = 2 * np.pi / wavelength
    pol_permit_ratio = 3 / Nm * (index_refraction**2 - 1) / (index_refraction**2 + 2)
    sigma = (8 * np.pi / 3) * (
        pol_permit_ratio * k_tweezer * k_tweezer / (4 * np.pi * epsilon_0)
    ) ** 2
    I0 = 2 * power / (np.pi * waist)
    return I0*sigma


def compute_backaction(wavelength, p_scat, A=0.71):
    k = 2 * np.pi / wavelength
    ba_force = np.sqrt(2 * (A**2 + 0.4) * hbar * k * p_scat / c)
    return ba_force


def compute_ideal_detection(wavelength, p_scat, A=0.71):
    k = 2 * np.pi / wavelength
    return np.sqrt(2 * hbar * c / ((A**2 + 0.4) * 4 * k * p_scat))

def compute_cavity_parameters(
    cavity_linewidth,
    cavity_length,
    detuning,
    wavelength
):
    cavity_freq = detuning + 2 * np.pi * c / wavelength
    phi = 4 * np.pi * cavity_freq * cavity_length / c
    FSR = c / (2 * cavity_length)
    f = FSR / cavity_linewidth
    '''(1-x)^2 = pi*x/f^2
    x^2 + 1 - x*(pi/f^2 + 2) = 0
    x = 1 + pi/2f^2 + 1/2*sqrt((pi/f^2 + 2)^2 - 4)'''
    r = 1 + np.pi/(2*f**2) - 1/2 * np.sqrt((2 + np.pi/f**2)**2-4)
    I_max = 1 / (1 - r)**2
    I_factor = I_max / (1 + (2 * f * np.sin(phi/2) / np.pi)**2)
    return FSR, f, r, I_factor
    
def compute_parameters_simulation(
    power,
    wavelength,
    tweezer_waist,
    omega,
    radius,
    pressure,
    fs,
    eta_detection,
    rho=2200,
    index_refraction=1.444,
    T=293,
):
    gamma = compute_gamma(
        radius,
        pressure,
        rho=rho,
        T=T,
    )
    p_scat = compute_scattered_power(
        power,
        tweezer_waist,
        wavelength,
        rho=rho,
        index_refraction=index_refraction,
    )
    ba_force = compute_backaction(wavelength, p_scat)
    std_z = compute_ideal_detection(wavelength, p_scat)
    std_detection = std_z * np.sqrt(fs / (2 * eta_detection))
    return gamma, ba_force, std_detection, std_z


def compute_parameters_simulation_cavity(
    power,
    wavelength,
    tweezer_waist,
    radius,
    pressure,
    fs,
    eta_detection,
    cavity_length,
    detuning_ratio,
    cavity_linewidth,
    omega,
    rho=2200,
    index_refraction=1.444,
    T=293,
):
    gamma = compute_gamma(
        radius,
        pressure,
        rho=rho,
        T=T,
    )
    '''omega = compute_omega(
        power,
        tweezer_waist,
        rho=rho,
        index_refraction=index_refraction,
    )'''
    p_scat = compute_scattered_power(
        power,
        tweezer_waist,
        wavelength,
        rho=rho,
        index_refraction=index_refraction,
    )
    ba_force = compute_backaction(wavelength, p_scat)
    std_z = compute_ideal_detection(wavelength, p_scat)
    std_detection = std_z * np.sqrt(fs / (2 * eta_detection))
    detuning = omega * detuning_ratio
    cavity_freq = detuning + 2 * np.pi * c / wavelength
    cavity_waist = np.sqrt(c*cavity_length / cavity_freq)
    V = 4/3 * np.pi * radius**3
    Vc = np.pi * cavity_waist**2 * cavity_length/4
    k = 2*np.pi/wavelength
    alpha = 3 * epsilon_0 * V * (index_refraction**2 - 1)/(index_refraction**2 + 2)
    E0 = np.sqrt(4 * power / (np.pi * tweezer_waist**2 * epsilon_0 * c))
    Ec = np.sqrt(hbar * cavity_freq / (2 * epsilon_0 * Vc))
    g_cs = 1/hbar * alpha * E0 * Ec * k * compute_zpx(omega, radius, rho = rho)

    '''g_cs[0] = (
        np.power(12 / np.pi, 1 / 4)
        * np.power((index_refraction**2 - 1) / (index_refraction**2 + 2), 3 / 4)
        * np.power(power * radius**6 * cavity_freq**6 / (c**5 * rho), 1 / 4)
        / (np.sqrt(cavity_length) * cavity_waist)
    )'''
    return (
        gamma,
        ba_force,
        std_detection,
        std_z,
        g_cs,
        detuning
    )
