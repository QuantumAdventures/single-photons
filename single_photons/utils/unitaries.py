# Gaussian unitaries (applicable to single mode states)
def displace(state, alpha, modes=[0]):
    state_copy = state.copy()
    state_copy.displace(alpha, modes)
    
    return state_copy

def squeeze(state, r, modes=[0]):
    state_copy = state.copy()
    state_copy.squeeze(r, modes)
    
    return state_copy

def rotate(state, theta, modes=[0]):
    state_copy = state.copy()
    state_copy.rotate(theta, modes)
    
    return state_copy 

def phase(state, theta, modes=[0]):
    state_copy = state.copy()
    state_copy.phase(theta, modes)
    
    return state_copy

def beam_splitter(state, tau, modes=[0, 1]):
    state_copy = state.copy()
    state_copy.beam_splitter(tau, modes)
    
    return state_copy

def two_mode_squeezing(state, r, modes=[0, 1]):
    state_copy = state.copy()
    state_copy.two_mode_squeezing(r, modes)
    
    return state_copy

# Generic multimode gaussian unitary
def apply_unitary(state, S, d):
    state_copy = state.copy()
    state_copy.apply_unitary(S, d)
    
    return state_copy
