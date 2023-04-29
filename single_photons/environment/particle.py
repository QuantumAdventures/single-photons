import numpy as np
T = 300.0 # Kelvin
kb = 1.38064852e-11 # picoJoule/Kelvin

class Particle:
    def __init__(self, omega, gamma, ):  
        self.__omega__ = omega
        self.__gamma__ = gamma
        self.A = np.array([[0, self.__omega__],
                           [-self.__omega__, -self.__gamma__]])
        self.B = np.array([[0], [1]])
        self.C = np.array([[1, 0]])

        
                 
        self._radius_ = radius #75e-3                     # particle radius         micrometers
        self._pressure_ = pressure
        self.T = T
        m_gas = 2.325e-26                  # nitrogen gas molecule   kg
        kb = 1.38064852e-11                # Boltzmann cst.          picoJoule/Kelvin
        v_gas = np.sqrt(3*kb*T/m_gas)      # meam squared velocity of nitrogen gas        micrometers/seconds
        self._gamma_ = 15.8*self._radius_**2*self._pressure_/(v_gas)
        self._rho_ = rho                    # silica density          kilogram/(micrometers)^3
        self._m_ = rho*4*np.pi*self._radius_**3/3
        self._k_ = omega*omega*self._m_
        self._omega_ = omega
        self._period_ = 2*np.pi/omega
        self.Q = elec_num*1.6e-19
        self.d = 8e-3
        self.A = np.array([[0, 1], [-self._k_/self._m_, -self._gamma_/self._m_]])
        self.B = np.array([[0],[self.Q/(self.d*self._m_)]])

    def step(self, states, control=0.0, delta_t=50e-2):
        if states.size > 2:
            raise ValueError('States size for this specific system is equal to two \
                (position and velocity)')
#        print(self.B*force)
        states_dot = np.matmul(self.A, states) + self.B*control
        v = states[1,0] + states_dot[1,0]*delta_t+ np.sqrt(2*kb*self._gamma_*self.T*delta_t) * np.random.normal()/self._m_
        x = states[0,0] + v*delta_t
        return x, v