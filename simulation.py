from spring_mass import SpringMass
import numpy as np
import matplotlib.pyplot as plt

def control(states, kp, kd):
    x, v = states.ravel()
    return -kp*x - kd*v

if __name__=='__main__':
    system = SpringMass(m=1, k=50, gamma=1)
    xs, vs = [], []
    states = np.array([[0], [2]])
    omega = np.sqrt(system._k_/system._m_)
    T = 2*np.pi/omega
    t = np.linspace(0, 5*T, 1000)
    for element in t:
        print(control(states, 1, 1))
        x, v = system.step(states=states, force=control(states, 1, 1), delta_t=np.diff(t)[0])
        states = np.array([[x], [v]])
        xs.append(x)
        vs.append(v)
    plt.plot(t, xs)
    plt.show()