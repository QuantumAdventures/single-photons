import numpy as np


class KalmanFilter:
     
    # x0 - initial guess of the state vector 
    # P0 - initial guess of the covariance matrix of the state estimation error
    # A,B,C - system matrices describing the system model
    # Q - covariance matrix of the process noise 
    # R - covariance matrix of the measurement noise
     
    def __init__(self,x0,P0,A,B,C,Q,R):
         
        # initialize vectors and matrices
        self.x0=x0
        self.P0=P0
        self.A=A
        self.B=B
        self.C=C
        self.Q=Q
        self.R=R
         
        # this variable is used to track the current time step k of the estimator 
        # after every measurement arrives, this variables is incremented for +1 
        self.t_i=0
         
        # this list is used to store the a posteriori estimates xk^{+} starting from the initial estimate 
        # note: list starts from x0^{+}=x0 - where x0 is an initial guess of the estimate
        self.estimates_aposteriori = []
        self.estimates_aposteriori.append(x0)
         
        # this list is used to store the a apriori estimates xk^{-} starting from x1^{-}
        # note: x0^{-} does not exist, that is, the list starts from x1^{-}
        self.estimates_apriori = []    
        # this list is used to store the a posteriori estimation error covariance matrices Pk^{+}
        # note: list starts from P0^{+}=P0, where P0 is the initial guess of the covariance
        self.error_covariance_aposteriori = []
        self.error_covariance_aposteriori.append(P0)
         
        # this list is used to store the a priori estimation error covariance matrices Pk^{-}
        # note: list starts from P1^{-}, that is, P0^{-} does not exist
        self.error_covariance_apriori = []
         
        # this list is used to store the gain matrices Kk
        self.gainMatrices=[]
          
        # this list is used to store prediction errors error_k=y_k-C*xk^{-}
        self.errors=[]
         
    # this function propagates x_{k-1}^{+} through the model to compute x_{k}^{-}
    # this function also propagates P_{k-1}^{+} through the covariance model to compute P_{k}^{-}
    # at the end this function increments the time index currentTimeStep for +1
    def propagateDynamics(self,inputValue):
         
        xk_minus=self.A*self.estimates_aposteriori[self.t_i]+self.B*inputValue
        Pk_minus=self.A*self.error_covariance_aposteriori[self.t_i]*(self.A.T)+self.Q
         
        self.estimates_apriori.append(xk_minus)
        self.error_covariance_apriori.append(Pk_minus)
         
        self.t_i=self.t_i+1
     
    # this function should be called after propagateDynamics() because the time step should be increased and states and covariances should be propagated         
    def computeAposterioriEstimate(self,currentMeasurement):
        # gain matrix
        Kk=self.error_covariance_apriori[self.t_i-1]*(self.C.T)*np.linalg.inv(self.R+self.C*self.error_covariance_apriori[self.t_i-1]*(self.C.T))
         
        # prediction error
        error_k=currentMeasurement-self.C*self.estimates_apriori[self.t_i-1]
        # a posteriori estimate
        xk_plus=self.estimates_apriori[self.t_i-1]+Kk*error_k
         
        # a posteriori matrix update 
        IminusKkC=np.matrix(np.eye(self.x0.shape[0]))-Kk*self.C
        Pk_plus=IminusKkC*self.error_covariance_apriori[self.t_i-1]*(IminusKkC.T)+Kk*(self.R)*(Kk.T)
         
        # update the lists that store the vectors and matrices
        self.gainMatrices.append(Kk)
        self.errors.append(error_k)
        self.estimates_aposteriori.append(xk_plus)
        self.error_covariance_aposteriori.append(Pk_plus)
