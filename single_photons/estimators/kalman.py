import numpy as np


class KalmanFilter:
     
    def __init__(self,x0,P0,A,B,C,Q,R):
        '''
          x0 - initial guess of the state vector 
         P0 - initial guess of the covariance matrix of the state estimation error
         A,B,C - system matrices describing the system model
         Q - covariance matrix of the process noise 
         R - covariance matrix of the measurement noise
        '''    
        self.x0=x0
        self.P0=P0
        self.A=A
        self.B=B
        self.C=C
        self.Q=Q
        self.R=R
        self.t_i=0
        self.estimates_aposteriori = [x0]
        self.estimates_apriori = []    
        self.error_covariance_aposteriori = [P0]
        self.error_covariance_apriori = []
        self.gainMatrices=[]
        self.errors=[]
         
    def propagate_dynamics(self,inputValue):
         
        xk_minus=self.A*self.estimates_aposteriori[self.t_i]+self.B*inputValue
        Pk_minus=self.A*self.error_covariance_aposteriori[self.t_i]*(self.A.T)+self.Q
        self.estimates_apriori.append(xk_minus)
        self.error_covariance_apriori.append(Pk_minus)         
        self.t_i=self.t_i+1
     
    def compute_aposteriori(self,currentMeasurement):
        Kk=self.error_covariance_apriori[self.t_i-1]*(self.C.T)*np.linalg.inv(self.R+self.C*self.error_covariance_apriori[self.t_i-1]*(self.C.T))
        error_k=currentMeasurement-self.C*self.estimates_apriori[self.t_i-1]
        xk_plus=self.estimates_apriori[self.t_i-1]+Kk*error_k
        IminusKkC=np.matrix(np.eye(self.x0.shape[0]))-Kk*self.C
        Pk_plus=IminusKkC*self.error_covariance_apriori[self.t_i-1]*(IminusKkC.T)+Kk*(self.R)*(Kk.T)
        self.gainMatrices.append(Kk)
        self.errors.append(error_k)
        self.estimates_aposteriori.append(xk_plus)
        self.error_covariance_aposteriori.append(Pk_plus)
