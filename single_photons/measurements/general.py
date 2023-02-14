def measurement_general(state, *args):
    state_copy = state.copy()
    state_copy.measurement_general(*args)
    
    return state_copy