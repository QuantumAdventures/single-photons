import numpy as np

from numba import njit, jit
from numba.pycc import CC


cc_c = CC("simulation_cavity")
cc_c._source_module = "single_photons.simulation.simulation_cavity"


@njit(nopython=True, cache=True)
@cc_c.export(
    "simulation_c",
    "Tuple((c16[:,:], c16[:,:], c16[:,:], c16[:,:,:], c16[:]))\
       (f8[:,:], f8[:,:], c16[:,:], f8, f8, f8, f8, f8, f8, f8[:,:], f8[:,:], \
       f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8, i8, i8)",
)
def simulation_c(
    A,
    B,
    optical_input,
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
    A = A.astype(np.complex_)
    B = B.astype(np.complex_)
    Ad = Ad.astype(np.complex_)
    Bd = Bd.astype(np.complex_)
    C = C.astype(np.complex_)
    G = G.astype(np.complex_)
    G_fb = G_fb.astype(np.complex_)
    Q = Q.astype(np.complex_)
    R = R.astype(np.complex_)
    kalman_array_size = 1
    for k in range(N_time):
        if not k % control_step:
            kalman_array_size += 1
    e_aposteriori = np.zeros((kalman_array_size + 1, 4, 1)).astype(np.complex_)
    estimation = np.array(
        [[0.0], [0.0], [x0 * np.random.normal()], [x0 * np.random.normal()]]
    ).astype(np.complex_)
    e_aposteriori[0, :, :] = estimation
    e_apriori = np.zeros((kalman_array_size, 4, 1)).astype(np.complex_)
    cov_aposteriori = np.zeros((kalman_array_size + 1, 4, 4)).astype(np.complex_)
    P0 = float(P0) * np.eye(4).astype(np.complex_)
    cov_aposteriori[0, :, :] = P0
    cov_apriori = np.zeros((kalman_array_size, 4, 4)).astype(np.complex_)
    kalman_gain_matrices = np.zeros((kalman_array_size, 4, 1)).astype(np.complex_)
    kalman_errors = np.zeros((kalman_array_size, 1, 1)).astype(np.complex_)
    state = np.zeros(shape=(N_time, 4)).astype(np.complex_)
    controls = np.zeros(shape=(N_time)).astype(np.complex_)
    measured_states = np.zeros(shape=(N_time, 1)).astype(np.complex_)
    estimated_states = np.zeros((N_time, 4)).astype(np.complex_)
    current_states = x0 * np.array(
        [
            [np.random.normal()],
            [np.random.normal()],
            [np.random.normal()],
            [np.random.normal()],
        ]
    ).astype(np.complex_)
    estimated_states[0, :] = estimation[:, 0]
    control = np.zeros((1, 1)).astype(np.complex_)
    kalman_time_step = 0
    for k in range(N_time):
        if not k % control_step:
            measured_states[k] = current_states[2, 0] + detect_std * np.random.normal()
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
                kalman_time_step,
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
            estimation = estimated_states[k, :].reshape((4, 1))
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
            + optical_input[:, k]
        )
        controls[k] = control[0, 0]
        state[k, :] = current_states[:, 0]
    return state, measured_states, estimated_states, cov_aposteriori, controls


@njit(nopython=True, cache=True)
@cc_c.export(
    "propagate_dynamics",
    "Tuple((c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:], i8))(c16[:,:], \
      c16[:,:], c16[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:], \
          c16[:,:], i8)",
)
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
    xk_minus = Ad @ e_aposteriori[time_step] + Bd * control
    Pk_minus = Ad @ (cov_aposteriori[time_step] @ (Ad.T)) + Q
    e_apriori[time_step] = xk_minus
    cov_apriori[time_step] = Pk_minus
    time_step = time_step + 1
    return e_aposteriori, e_apriori, cov_aposteriori, cov_apriori, time_step


@njit(nopython=True, cache=True)
@cc_c.export(
    "compute_aposteriori",
    "Tuple((c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:]))(c16[:,:], \
      c16[:,:], c16[:,:], c16[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:],\
            c16[:,:,:], c16[:,:,:], c16[:,:,:], i8)",
)
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
    Kk = cov_apriori[time_step - 1] @ (
        C.T @ np.linalg.pinv(R + C @ (cov_apriori[time_step - 1] @ (C.T)))
    )
    error_k = measurement - C @ e_apriori[time_step - 1]
    xk_plus = e_apriori[time_step - 1] + Kk @ error_k
    IminusKkC = np.eye(estimation.shape[0]) - Kk @ C
    Pk_plus = IminusKkC @ (cov_apriori[time_step - 1] @ (IminusKkC.T)) + Kk @ (R @ Kk.T)
    kalman_gain_matrices[time_step] = Kk
    kalman_errors[time_step] = error_k
    e_aposteriori[time_step] = xk_plus
    cov_aposteriori[time_step] = Pk_plus
    return e_aposteriori, cov_aposteriori, kalman_errors, kalman_gain_matrices


if __name__ == "__main__":
    cc_c.compile()
