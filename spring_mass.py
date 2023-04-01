import numpy as np

class SpringMass:
    def __init__(self, m, k, gamma):
        self._m_ = m
        self._k_ = k
        self._gamma_ = gamma
        self.A = np.array([[0, 1], [-k/m, -gamma/m]])
        self.B = np.array([[0],[1]])

    def step(self, states, force=0.0, delta_t=50e-2):
        if states.size > 2:
            raise ValueError('States size for this specific system is equal to two \
                (position and velocity)')
        states_dot = np.matmul(self.A, states) + self.B*force
        v = states[1,0] + states_dot[1,0]*delta_t
        x = states[0,0] + v*delta_t
        return x, v