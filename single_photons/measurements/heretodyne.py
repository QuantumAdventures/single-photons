def measurement_heterodyne(state, *args):
    state_copy = state.copy()
    state_copy.measurement_heterodyne(*args)
    
    return state_copy