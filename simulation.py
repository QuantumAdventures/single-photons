#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:16:38 2023

@author: rafaelmuffatoreis
"""


from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from scipy import signal as sn
from scipy.optimize import curve_fit
import csv
from numba import jit


radius = 75e-3                     # particle radius         micrometers
presure = 1e-3                     # gas pressure            kg/(micrometer*second^2)
m_gas = 2.325e-26                  # nitrogen gas molecule   kg
T = 300.0                            # temperature             Kelvin
kb = 1.38064852e-11                # Boltzmann cst.          picoJoule/Kelvin
v_gas = np.sqrt(3*kb*T/m_gas)      # meam squared velocity of nitrogen gas        micrometers/seconds
gamma = 15.8*radius**2*presure/(v_gas)
rho = 2200*1e-18                   # silica density          kilogram/(micrometers)^3
massa = rho*4*np.pi*radius**3/3    # mass                    kg
n_m = 1.01                         # medium refractive index
n_p = 1.46                         # particle refractive 
m = (n_p/n_m)                      # relative refractive  
NA = 0.7                           # numerical aperture
c = 3e14                           # speed of light          micrometers/seconds
P = 50e9                           # power                   kilogram*micrometers^2/seconds^3
wl0 = 0.78                         # laser wavelength        micrometers
wl_m = wl0/n_m                     # wavelength              micrometers
w0 = wl_m/(np.pi*NA)               # beam waist              micrometers
zr = np.pi*w0**2/wl_m              # railegh range           micrometers
I0 = 2*P/(np.pi*w0**2)             # intensity               Watts/meter^2
V0 = -(2*np.pi*n_m*radius**3/c)*((m**2-1)/(m**2+2))*I0 
                                   # potential depth         picoJoule 
spring = -2*V0/(zr**2)             # spring constant         microNewtons/micometers
t_relaxation = gamma/massa         # relaxation time         seconds
t_period =2*np.pi*np.sqrt(massa/spring)

max_time = 1600*t_period              # seconds
dt = t_period/400                  # numerical integration with 400 points per period 
reduction = 100                    # one useful state point at avery few integration points
f_integration = 1/dt
f_sampling = f_integration/reduction

f_resonance = 1/t_period


N_time = int(max_time/dt)          # size of simulation

N_simulation = int(N_time/reduction)
psd_stamps =int( N_time/(reduction*2)+1)       # size of periodogram frequency array span
t = np.linspace(0,max_time,int(N_time/reduction))/t_period # time                    seconds


x0 = np.sqrt(kb*T/spring)          # length scale
v0 = np.sqrt(kb*T*spring)/gamma
# f0 = np.sqrt(spring*kb*T)

perturbation_ratio_amoun = 0.1
ratio = np.linspace(-perturbation_ratio_amoun,perturbation_ratio_amoun,3)

for k, perturbation_ratio in enumerate(ratio):



    # def electric_force(z,perturbation_ratio):
    #     elec_number = 20.0
    #     elec_charge = 1.6e-19                        # electron charge
    #     # perturbation_ratio = 0.005
    #     charge = elec_number*elec_charge
    #     E0 = 2*spring*x0*perturbation_ratio/(charge)
    #     E = charge*E0*(z/x0)**3
    #     return charge*E
    
    @jit(nopython=True)
    def simulation(eee):
        elec_number = 20.0
        elec_charge = 1.6e-19                        # electron charge
        # perturbation_ratio = 0.005
        charge = elec_number*elec_charge
        E0 = 2*spring*x0*perturbation_ratio/(charge)
        
        state = np.zeros(shape= (N_simulation,2))
    
        
        v = 0
        x = 0
        printcounter = 0
        for k in range(N_time-1):
            v = v - (gamma/massa)*v*dt -(spring/massa)*x*dt - charge*E0*(x/x0)**3*dt/massa + np.sqrt(2.0 * kb * gamma * T * dt ) * np.random.normal()/massa # Numerical integration of velocity
            x = x +v*dt                                                                                           # numerical integration of position
           
            
            if (printcounter == reduction):  # Storing less data than used to integrate.
                state[int(k/reduction),1] = v
                state[int(k/reduction),0] = x
                printcounter = 0
            printcounter += 1
        
        state[:,0] = state[:,0]
            
        return state[:,0]
    
    processes = 6
    M = 100
    storage = np.zeros(shape = (M, processes, psd_stamps))
    
    def psd(eee):
        state = simulation(eee)
        freq, pxx = sn.periodogram(state,f_sampling)
        return pxx
    
    def frequencies(eee):
        state = simulation(eee)
        freq, pxx = sn.periodogram(state,f_sampling)
        return freq
    
    def lorentzian(omega,A,B,C):
        num = A
        den = (B**2-omega**2)**2+omega**2*C**2
        return num/den
    
    def main():
        if __name__ == '__main__':
            p = Pool(processes)
            for i in tqdm(range(M)):
                data = p.map(psd,range(processes))
                data = np.array(data)
                storage[i,:,:] = data

            # storage_reshaped = np.reshape(storage,(500,16,psd_stamps))
            smooth_data = np.mean(storage,axis = 0)
            np.save("file"+str(k)+".npy",smooth_data)
            

            # psd_mean = np.mean(smooth_data,axis = 0)[8000-400:8000+400]
            # psd_std = np.std(smooth_data,axis = 0)[8000-400:8000+400]
            # freq = frequencies(0)[8000-400:8000+400]
    
    
    

            # p0 = [10000,f_resonance,1000]
            # ans, cov = curve_fit(lorentzian,freq,psd_mean, p0 = p0,sigma = psd_std, absolute_sigma=True)
           
            # center_frequency = ans[1]
            # cf_std = np.sqrt(cov[1,1])
            # myCsvRow = [(perturbation_ratio, center_frequency, cf_std)]
            # with open('file.csv', 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(myCsvRow)
 

    
    main()