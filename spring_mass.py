import numpy as np
T = 300.0 # Kelvin
kb = 1.38064852e-11 # picoJoule/Kelvin

class TweezedParticle:
    def __init__(self, radius, pressure, omega, T=300, rho=2200*1e-18):
        self._radius_ = radius #75e-3                     # particle radius         micrometers
        self._pressure_ = pressure
        self.T = T
        m_gas = 2.325e-26                  # nitrogen gas molecule   kg
        kb = 1.38064852e-11                # Boltzmann cst.          picoJoule/Kelvin
        v_gas = np.sqrt(3*kb*T/m_gas)      # meam squared velocity of nitrogen gas        micrometers/seconds
        self._gamma_ = 15.8*self._radius_**2*self._pressure_/(v_gas)
        print(self._gamma_)
        self._rho_ = rho                    # silica density          kilogram/(micrometers)^3
        self._m_ = rho*4*np.pi*self._radius_**3/3  
        self._k_ = omega*omega*self._m_
        self.A = np.array([[0, 1], [-self._k_/self._m_, -self._gamma_/self._m_]])
        self.B = np.array([[0],[1]])

    def step(self, states, force=0.0, delta_t=50e-2):
        if states.size > 2:
            raise ValueError('States size for this specific system is equal to two \
                (position and velocity)')
        states_dot = np.matmul(self.A, states) + self.B*force + np.sqrt(2*kb*self._gamma_*self.T*delta_t) * np.random.normal()/self._m_
        v = states[1,0] + states_dot[1,0]*delta_t
        x = states[0,0] + v*delta_t
        return x, v