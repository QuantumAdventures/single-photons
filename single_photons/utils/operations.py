def tensor_product(rho_list):
    state_copy = rho_list[0].copy()
    state_copy.tensor_product(rho_list[1:])
    
    return state_copy

def partial_trace(state, indexes):
    state_copy = state.copy()
    state_copy.partial_trace(indexes)
    
    return state_copy

def only_modes(state, indexes):
    state_copy = state.copy()
    state_copy.only_modes(indexes)
    
    return state_copy

def check_uncertainty_relation(state):
    return state.check_uncertainty_relation()

def loss_ancilla(state,idx,tau):
    state_copy = state.copy()
    state_copy.loss_ancilla(state,idx,tau)
    
    return state_copy


# Properties of a gaussian state