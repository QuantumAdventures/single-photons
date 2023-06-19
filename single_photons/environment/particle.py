import numpy as np
import single_photons.utils.constants as ct


class Particle:
    def __init__(
        self, omega, gamma, coupling, eta_detection=1, radius=147e-9, rho=2200, T=293
    ):
        self.__omega__ = omega
        self.__gamma__ = gamma
        self.T = T
        self.A = np.array([[0, self.__omega__], [-self.__omega__, -self.__gamma__]])
        self.B = np.array([[0], [1]])
        
        self.G = np.array([[0], [1]])
        self.backaction = np.sqrt(4 * np.pi * coupling)
        self.eta_det = eta_detection
        self._m_ = rho * 4 * np.pi * np.power(radius, 3) / 3
        self.zp_x = np.sqrt(ct.hbar / (2 * omega * self._m_))
        self.zp_p = np.sqrt(omega * ct.hbar * self._m_ / 2)
        self.nl = ct.hbar*self.__omega__/(ct.kb*self.T)
        self.C = np.array([[16*np.pi*coupling*eta_detection, 0]])/self.x_zp
        self.thermal_force_std = np.sqrt(4 * self.__gamma__ * self._m_ * ct.kb * T)/self.zp_p
        self.backaction_std = self.backaction/ self.zp_p

    def __backaction_fluctuation__(self):
        return self.backaction_std * (
            np.sqrt(self.eta_det) * np.random.normal()
            + np.sqrt(1 - self.eta_det) * np.random.normal()
        )

    def step(self, states, control=0.0, delta_t=50e-2):
        if states.size > 2:
            raise ValueError(
                "States size for this specific system is equal to two \
                (position and velocity)"
            )
        backaction_force = self.__backaction_fluctuation__()
        thermal_force = self.thermal_force_std* np.random.normal()
        state_dot = np.matmul(self.A, states) + self.B * control
        states = (
            states
            + state_dot * delta_t
            + self.G * np.sqrt(delta_t) * (thermal_force - backaction_force)
        )
        return states
