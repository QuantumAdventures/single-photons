import numpy as np
from numba import njit, jit
from numba.pycc import CC


# cc_p = CC("simulation_particle")
# cc_p._source_module = "single_photons.simulation.simulation_particle"

"""
@njit(nopython=True, cache=True)
@cc_p.export(
    "simulation_p",
    "Tuple((f8[:,:], f8[:,:], f8[:,:], f8[:,:,:], f8[:]))\
       (f8[:,:], f8[:,:], f8, f8, f8, f8, f8, f8, f8[:,:], f8[:,:], \
       f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8, i8, i8)",
)
"""


@jit(nopython=True)
def simulation_p(
    A,
    B,
    thermal_std,
    backaction_std,
    detect_std,
    eta_det,
    x0,
    P0,
    Ad,
    Bd,
    C,
    G,
    G_fb,
    Q,
    R,
    dt,
    control_step,
    N_time,
):
    kalman_array_size = 1
    for k in range(N_time):
        if not k % control_step:
            kalman_array_size += 1
    e_aposteriori = np.zeros((kalman_array_size + 1, 2, 1))
    estimation = np.array([[x0 * np.random.normal()], [x0 * np.random.normal()]])
    e_aposteriori[0, :, :] = estimation
    e_apriori = np.zeros((kalman_array_size, 2, 1))
    cov_aposteriori = np.zeros((kalman_array_size + 1, 2, 2))
    P0 = float(P0) * np.eye(2)
    cov_aposteriori[0, :, :] = P0
    cov_apriori = np.zeros((kalman_array_size, 2, 2))
    kalman_gain_matrices = np.zeros((kalman_array_size, 2, 1))
    kalman_errors = np.zeros((kalman_array_size, 1, 1))
    state = np.zeros(shape=(N_time, 2))
    controls = np.zeros(shape=(N_time))
    measured_states = np.zeros(shape=(N_time, 1))
    estimated_states = np.zeros((N_time, 2))
    current_states = np.array([[x0 * np.random.normal()], [x0 * np.random.normal()]])
    estimated_states[0, :] = estimation[:, 0]
    control = np.zeros((1, 1))
    kalman_time_step = 0
    for k in range(N_time):
        if not k % control_step:
            measured_states[k] = current_states[0, 0] + detect_std * np.random.normal()
            (
                e_aposteriori,
                e_apriori,
                cov_aposteriori,
                cov_apriori,
                kalman_time_step,
            ) = propagate_dynamics(
                Ad,
                Bd,
                Q,
                e_aposteriori,
                e_apriori,
                cov_aposteriori,
                cov_apriori,
                control,
                kalman_time_step
            )
            (
                e_aposteriori,
                cov_aposteriori,
                kalman_errors,
                kalman_gain_matrices,
            ) = compute_aposteriori(
                measured_states[k],
                C,
                R,
                estimation,
                e_aposteriori,
                e_apriori,
                cov_aposteriori,
                cov_apriori,
                kalman_gain_matrices,
                kalman_errors,
                kalman_time_step,
            )
            estimated_states[k, :] = e_aposteriori[int(k / control_step), :, 0]
            estimation = estimated_states[k, :].reshape((2, 1))
            control = -G_fb @ estimation
        else:
            measured_states[k] = measured_states[k - 1]
            estimated_states[k, :] = estimated_states[k - 1, :]
        state_dot = A @ current_states + B * control
        backaction_term = backaction_std * (
            np.sqrt(eta_det) * np.random.normal()
            + np.sqrt(1 - eta_det) * np.random.normal()
        )
        current_states = (
            current_states
            + state_dot * dt
            + G * np.sqrt(dt) * (backaction_term + thermal_std * np.random.normal())
        )
        controls[k] = control[0, 0]
        state[k, :] = current_states[:, 0]
    return state, measured_states, estimated_states, cov_aposteriori, controls


"""
@njit(nopython=True, cache=True)
@cc_p.export(
    "propagate_dynamics",
    "Tuple((f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], i8))(f8[:,:], \
       f8[:,:], f8[:,:], f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:], \
           f8[:,:], i8)",
)"""


@jit(nopython=True)
def propagate_dynamics(
    Ad,
    Bd,
    Q,
    e_aposteriori,
    e_apriori,
    cov_aposteriori,
    cov_apriori,
    control,
    time_step,
):
    xk_minus = Ad @ e_aposteriori[time_step] + Bd @ control
    Pk_minus = Ad @ (cov_aposteriori[time_step] @ (Ad.T)) + Q
    e_apriori[time_step] = xk_minus
    cov_apriori[time_step] = Pk_minus
    time_step = time_step + 1
    return e_aposteriori, e_apriori, cov_aposteriori, cov_apriori, time_step


"""
@njit(nopython=True, cache=True)
@cc_p.export(
    "compute_aposteriori",
    "Tuple((f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:]))(f8[:,:], \
       f8[:,:], f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:],\
             f8[:,:,:], f8[:,:,:], f8[:,:,:], i8)",
)
"""


@jit(nopython=True)
def compute_aposteriori(
    measurement,
    C,
    R,
    estimation,
    e_aposteriori,
    e_apriori,
    cov_aposteriori,
    cov_apriori,
    kalman_gain_matrices,
    kalman_errors,
    time_step,
):
    '''error_k = measurement - C @ e_apriori[time_step - 1]
    s_k = C @ cov_apriori[time_step-1] @ C.T + R #
    Kk = cov_apriori[time_step-1] @ C.T @ np.linalg.pinv(s_k) #
    xk_plus = e_apriori[time_step-1] + Kk @ error_k
    Pk_plus = cov_apriori[time_step-1] - Kk @ s_k @ Kk.T#'''
    
    error_k = measurement - C @ e_apriori[time_step - 1]
    Kk = cov_apriori[time_step - 1] @ (
        C.T @ np.linalg.pinv(R + C @ (cov_apriori[time_step - 1] @ (C.T)))
    )
    IminusKkC = np.eye(estimation.shape[0]) - Kk @ C
    xk_plus = e_apriori[time_step-1] + Kk @ error_k
    Pk_plus = IminusKkC @ (cov_apriori[time_step - 1] @ (IminusKkC.T)) + Kk @ (R @ Kk.T)
    
    kalman_gain_matrices[time_step] = Kk
    kalman_errors[time_step] = error_k
    e_aposteriori[time_step] = xk_plus
    cov_aposteriori[time_step] = Pk_plus
    return e_aposteriori, cov_aposteriori, kalman_errors, kalman_gain_matrices


# if __name__ == "__main__":
#    cc_p.compile()
