def measurement_homodyne(state, *args):
    state_copy = state.copy()
    state_copy.measurement_homodyne(*args)
    
    return state_copy
