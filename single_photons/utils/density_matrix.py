# Density matrix elements
def density_matrix_coherent_basis(state, alpha, beta):
    return state.coherence(alpha, beta)

def density_matrix_number_basis(state, n_cutoff=10):
    return state.density_matrix_number_basis(n_cutoff)

def number_statistics(state, n_cutoff=10):
    return state.number_statistics(n_cutoff)