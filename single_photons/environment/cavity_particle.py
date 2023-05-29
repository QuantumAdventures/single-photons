import numpy as np
import single_photons.utils.constants as ct


class Cavity_Particle:
    def __init__(self, omega_p, gamma, detuning, kappa, g_cs, coupling, eta_detection=1, radius=147e-9, rho=2200):  
        self.__omega_p__ = omega_p
        self.__gamma__ = gamma
        self.__detuning__ = detuning
        self.__kappa__ = kappa
        self.__g_cs__ = g_cs
        self.A = np.array([[-self.__kappa__/2, self.__detuning__, 0, 0],
                           [-self.__detuning__, -self.__kappa__/2, -2*self.__g_cs__, 0],
                           [0, 0, 0, self.__omega_p__],
                           [-2*self.__g_cs__, 0, -self.__omega_p__, -self.__gamma__]])
        self.B = np.array([[0], [0], [0], [1]])
        self.C = np.array([[0, 0, 1, 0]])
        self.G = np.array([[0], [0], [0], [1]])
        self.backaction = np.sqrt(4*np.pi*coupling)
        self.eta_det = eta_detection
        self._m_ = rho*4*np.pi*np.power(radius, 3)/3
        self.zp_x = np.sqrt(ct.hbar/(omega_p*self._m_))
        self.zp_p = np.sqrt(omega_p*ct.hbar*self._m_)

    def __backaction_fluctuation__(self):
        return self.backaction*(np.sqrt(self.eta_det)*np.random.normal()+np.sqrt(1-self.eta_det)*np.random.normal())

    def step(self, states, alpha_in = 0, control=0.0, delta_t=50e-2):
        if states.size > 4:
            raise ValueError('States size for this specific system is equal to four \
                (two optical quadratures, position and velocity)')
        backaction_force = self.__backaction_fluctuation__()
        thermal_force = np.sqrt(2*self.__gamma__)*np.random.normal()
        x_in = np.sqrt(self.__kappa__)*(np.conjugate(alpha_in) + alpha_in)
        y_in = 1j*np.sqrt(self.__kappa__)*(np.conjugate(alpha_in) - alpha_in)
        optical_input = np.array([[x_in], [y_in], [0], [0]])
        state_dot = np.matmul(self.A,states) + self.B*control
        states = states + state_dot*delta_t + self.G*np.sqrt(delta_t)*(thermal_force+backaction_force) + optical_input
        return states