import numpy as np
import scipy
import numpy.linalg as linalg


def compute_times(signal, time_step):
    if max(abs(signal)) != max(signal):
        signal = -signal
    idx_max = 0
    M = 0
    mean = np.mean(signal[:int(signal.shape[0]/10)])
    for idx in range(signal.shape[0]):
        if signal[idx] > M:
            idx_max = idx
            M = signal[idx]
    idx_left = idx_max
    idx_right = idx_max
    while 1:
        c = 0
        if signal[idx_left] > mean + (M-mean)/np.e:
            idx_left = idx_left - 1
            c = 1
        if signal[idx_right] > mean + (M-mean)/np.e:
            idx_right = idx_right + 1
            c = 1
        if c == 0:
            break
        if idx_left < 0 or idx_right >= signal.shape[0]:
            print('Error with calculation of times')
            return None
    rising = (idx_max-idx_left)*time_step
    acommodation = (idx_right-idx_max)*time_step
    return rising, acommodation, idx_left, idx_right


def compute_SNR(signal, left, right):
    crop = signal[left:right]
    size = crop.shape[0]
    size = int(min(size, left))
    reference = signal[:size]
    SNR = np.mean(np.square(crop)) / np.mean(np.square(reference))
    return SNR


def compute_phonons(estimations, cov_matrix, control_step, step=30):
    estimates = estimations[::control_step]
    sampled_cov_matrix = cov_matrix[:, :, :estimates.shape[0]]
    phonons = np.zeros(int(estimates.shape[0]/step)-1)
    for i in range(1, int(estimates.shape[0]/step)):
        averaged = estimates[(i-1)*step:i*step, 0:].mean(axis=0)
        second_moments = sampled_cov_matrix[i]+np.power(averaged, 2)
        phonons[i-1] = np.trace(second_moments)/4-0.5
    return phonons


def compute_F0(V1, V2):
    symp = np.array([[0, 1, 0, 0],
                     [-1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, -1, 0]])
    identity = np.eye(4)
    V_aux = np.matmul(np.matmul(np.transpose(symp), np.linalg.pinv(V1+V2)),
                      symp/4 + np.matmul(np.matmul(V2, symp), V1))
    aux_inv = linalg.pinv(np.matmul(V_aux, symp))
    aux_powered = linalg.matrix_power(aux_inv, 2)/4
    F0 = np.power(
        np.linalg.det(
            2*np.matmul(
                scipy.linalg.fractional_matrix_power(
                    identity + aux_powered, 0.5
                    ) + identity, V_aux
                    )
                    ), 1/4
                )
    F0 = F0/np.power(np.linalg.det(V1+V2), 1/4)
    return F0


def compute_fidelity(estimations, cov_matrix, cov_ss, control_step, step=30):
        aux = estimations[::control_step]
        N = int(aux.shape[0]/step)
        estimates = []
        sampled_cov_matrix = []
        for i in range(N):
            mean = np.mean(aux[step*i:step*(i+1)-1], axis=0)
            averaged = aux[i*step:step*(i+1)-1, 0:].mean(axis=0)
            cov = np.mean(cov_matrix[step*i:step*(i+1)-1], axis=0)
            estimates.append(mean)
            sampled_cov_matrix.append(cov + np.diag(np.power(averaged, 2)))
        estimates = np.array(estimates)
        sampled_cov_matrix = np.array(sampled_cov_matrix)
        N = len(sampled_cov_matrix)
        fidelity = np.zeros(N).astype(np.complex_)
        for i in range(N):
            du = estimates[i]
            V = np.linalg.pinv(sampled_cov_matrix[i] + cov_ss)
            fidelity[i] = compute_F0(sampled_cov_matrix[i], cov_ss) *\
                np.exp(-1/4*np.matmul(np.matmul(np.transpose(du), V), du))
        return max(fidelity)/fidelity
