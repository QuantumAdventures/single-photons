import numpy as np
import matplotlib.pyplot as plt
from spring_mass import SpringMass 


def harmonic_oscillator_energy_force(x,k,x0):
    #calculate harmonic force
    force = -k*(x-x0)
    return force


def position_update(x,v,dt):
    x_new = x + v*dt/2.
    return x_new


def velocity_update(v,F,dt):
    v_new = v + F*dt/2.
    return v_new


def random_velocity_update(v,gamma,kBT,dt):
    R = np.random.normal()
    c1 = np.exp(-gamma*dt)
    c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT)
    v_new = c1*v + R*c2
    return v_new


def baoab(potential, max_time, dt, gamma, kBT, initial_position, initial_velocity,
                                        save_frequency=3, **kwargs ):
    x = initial_position
    v = initial_velocity
    t = 0
    step_number = 0
    positions = []
    velocities = []
    total_energies = []
    save_times = []
    
    while(t<max_time):
        
        # B
        force = potential(x,k,x0)
        v = velocity_update(v,force,dt)
        
        #A
        x = position_update(x,v,dt)

        #O
        v = random_velocity_update(v,gamma,kBT,dt)
        
        #A
        x = position_update(x,v,dt)
        
        # B
        force = potential(x,k,x0)
        v = velocity_update(v,force,dt)
        
        if step_number%save_frequency == 0 and step_number>0:

            positions.append(x)
            velocities.append(v)
            save_times.append(t)
        
        t = t+dt
        step_number = step_number + 1
    
    return save_times, positions, velocities, total_energies   

# Max # iterations in the simulation 
time = 20
# Initial conditions
x0 = 0.0
v0 = 0.1
# Model parameters

k = 20.
gamma= 0.2
kBT= 0.2
dt= 0.001

# measurement precision parameter
eta = 0.1


# Run simulation
times, positions, velocities, total_energies = baoab(harmonic_oscillator_energy_force, time, dt, gamma, kBT, x0, v0,k=k)

# Define the Kalman filter matrices

A = np.array([[0,np.sqrt(k)],[-np.sqrt(k),-gamma]]) * dt

# Discretized time evolution operator
Ad = np.array([[1,0],[0,1]]) + A + 0.5 * A.dot(A)

# Covariance of Brownian motion noise
Q = np.array([[0,0],[0,np.sqrt(2*gamma*kBT * dt)]])

# Covariance of measurement noise
meas_var = (eta**2)*(2*gamma*kBT)
R = np.array([[meas_var,0],[0,meas_var]])

# Measurement matrix
#C = np.array([[1,0],[0,1]])
C = np.array([[1,0],[0,0]])

# We consider C = Identity

# Define initial state and covariance matrix
position = np.array([x0,v0])


err = 1e-2
sigma = np.array([[err,0],[0,err]])

# Estimates array
position_estimate = []
sigma_estimate = []
measurement_data = []

# start main loop

for i in range(len(times)):
    
    #New measurement
    v = np.sqrt(meas_var) * np.random.normal()
    x_meas = np.array([positions[i], velocities[i]])
    measurement_noise = np.array([v,v])
    measurement = C.dot(x_meas)
    z = measurement + measurement_noise
    measurement_data.append(z[0])
    
    # Compute Kalman gain matrix
    M1 =  Ad.dot(sigma)
    M2 = M1.dot(Ad.transpose())
    sigma_step = M2 + Q
    
    N1 = C.dot(sigma_step)
    N2 = N1.dot(C.transpose())
    N3 = R + N2
    N4 = np.linalg.pinv(N3)
    
    N5 = sigma_step.dot(C.transpose())
    
    # Kalman gain
    K = N5.dot(N4)
    
    # Update covariance
    N6 = K.dot(C)
    sigma = sigma_step - N6.dot(sigma_step)
    
    # update position
    N7 = C.dot(Ad)
    position = Ad.dot(position) +  K.dot(z - N7.dot(position))
    
    position_estimate.append(position[0])
    sigma_estimate.append(sigma[0,0])
    
    

positions_plus  = [x+y for x,y in zip(position_estimate, sigma_estimate)]
positions_minus  = [x-y for x,y in zip(position_estimate, sigma_estimate)]



plt.figure()
#plt.plot(times,velocities,marker='',label='velocity',linestyle='-')
plt.plot(times,measurement_data,'-',color='k',alpha=0.2,linewidth=1.0,label='Measurement data')
plt.plot(times,positions_plus,'-',color='1',alpha=0.4)
plt.plot(times,positions_minus,'-',color='1',alpha=0.4)
plt.fill_between(times, positions_minus, positions_plus,color='C0',alpha=0.2)
plt.plot(times,positions,label='position',linestyle='-',color='k',alpha=0.9, linewidth=1.)
plt.plot(times,position_estimate,'-',color='C0',alpha=0.8,linewidth=1.,label='Kalman estimate')
plt.xlabel('time')
plt.legend(loc='upper right')
plt.show()
