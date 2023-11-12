import numpy as np
from control import lqr
import single_photons.utils.constants as ct


class Cavity_Particle:
    def __init__(
        self,
        omega_p,
        gamma,
        detuning,
        kappa,
        g_cs,
        coupling,
        eta_detection=1,
        radius=147e-9,
        rho=2200,
        T=293,
        g_fb_ratio=5,
    ):
        self.__omega_p__ = omega_p
        self.__gamma__ = gamma
        self.__detuning__ = detuning
        self.__kappa__ = kappa
        self.__g_cs__ = g_cs
        self.__shot_std__ = np.sqrt(self.__kappa__/2)
        self.A = np.array(
            [
                [-self.__kappa__ / 2, self.__detuning__, 0, 0],
                [-self.__detuning__, -self.__kappa__ / 2, -2 * self.__g_cs__, 0],
                [0, 0, 0, self.__omega_p__],
                [-2 * self.__g_cs__, 0, -self.__omega_p__, -self.__gamma__],
            ]
        )
        self.B = np.array([[0], [0], [0], [1.0]]).astype(float)
        self.C = np.array([[0, 0, 1, 0]]).astype(float)
        self.G = np.array([[0], [0], [0], [1.0]]).astype(float)
        self.T = T
        self.backaction = np.sqrt(4 * np.pi * coupling)
        self.eta_det = eta_detection
        self._m_ = rho * 4 * np.pi * np.power(radius, 3) / 3
        self.zp_x = np.sqrt(ct.hbar / (2 * omega_p * self._m_))
        self.zp_p = np.sqrt(omega_p * ct.hbar * self._m_ / 2)
        self.thermal_force_std = (
            np.sqrt(4 * self.__gamma__ * self._m_ * ct.kb * T) / self.zp_p
        )
        self.backaction_std = self.backaction / self.zp_p
        self.cost_states = np.array(
            [
                [self.__omega_p__ / 2, 0, 0, 0],
                [0, self.__omega_p__ / 2, 0, 0],
                [0, 0, self.__omega_p__ / 2, 0],
                [0, 0, 0, self.__omega_p__ / 2],
            ]
        )
        self.g_fb = g_fb_ratio * self.__omega_p__
        self.G_lqr = lqr(
            self.A, self.B, self.cost_states, self.__omega_p__ / (self.g_fb) ** 2
        )[0]

    def __backaction_fluctuation__(self):
        return self.backaction_std * (
            np.sqrt(self.eta_det) * np.random.normal()
            + np.sqrt(1 - self.eta_det) * np.random.normal()
        )

    def step(self, states, alpha_in=0, control=0.0, delta_t=50e-2):
        if states.size > 4:
            raise ValueError(
                "States size for this specific system is equal to four \
                (two optical quadratures, position and velocity)"
            )
        backaction_force = self.__backaction_fluctuation__()
        thermal_force = self.thermal_force_std * np.random.normal()
        x_in = (
            np.sqrt(self.__kappa__)
            * delta_t
            * (
                + np.conjugate(alpha_in)
                + alpha_in
            )
        )
        y_in = (
            1j
            * np.sqrt(self.__kappa__)
            * delta_t
            * (
                + np.conjugate(alpha_in)
                - alpha_in
            )
        )
        optical_noise = np.array([
            [self.__shot_std__*np.random.normal()],
            [self.__shot_std__*np.random.normal()],
            [0],
            [0]
        ])
        optical_input = np.array([[x_in], [y_in], [0], [0]])
        state_dot = np.matmul(self.A, states) + self.B * control
        states = (
            states
            + state_dot * delta_t
            + self.G * np.sqrt(delta_t) * (thermal_force - backaction_force)
            + optical_input+np.sqrt(delta_t)*optical_noise
        )
        return states
