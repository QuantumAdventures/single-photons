def symplectic_eigenvalues(state):
    return state.symplectic_eigenvalues()

def purity(state):
    return state.purity()

def squeezing_degree(state):
    return state.squeezing_degree()

def von_Neumann_Entropy(state):
    return state.von_Neumann_Entropy()

def mutual_information(state):
    return state.mutual_information()

def occupation_number(state):
    return state.occupation_number()

def number_operator_moments(state):
    return state.number_operator_moments()

def coherence(state):
    return state.coherence()

def logarithmic_negativity(state, *args):
    return state.logarithmic_negativity(*args)

def fidelity(rho_1, rho_2):
    return rho_1.fidelity(rho_2)


