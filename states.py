import numpy as np

from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix, random_statevector, random_density_matrix

def zero_state(n: int, verbose=False):
    """Generate a zero state.

    Args:
    n (int): Number of qubits.
    verbose (bool, optional): If True, print the generated state. Default is False.

    Returns:
    Statevector: A zero state.
    """

    zero_st = Statevector.from_label('0'*n)
    if verbose: print(zero_st)
    return zero_st

def plus_state(n: int, verbose=False):
    """Generate a plus state.

    Args:
    n (int): Number of qubits.
    verbose (bool, optional): If True, print the generated state. Default is False.

    Returns:
    Statevector: A plus state.
    """

    plus_st = (Statevector.from_label('0'*n) + Statevector.from_label('1'*n))/np.sqrt(2)
    if verbose: print(plus_st)
    return plus_st

def ghz_state(n: int, verbose=False):
    """Generate a GHZ state.

    Args:
    n (int): Number of qubits.
    verbose (bool, optional): If True, print the generated state. Default is False.

    Returns:
    Statevector: A GHZ state.
    """

    ghz_st = (Statevector.from_label('0'*n) + Statevector.from_label('1'*n))/np.sqrt(2)
    if verbose: print(ghz_st)
    return ghz_st
    ## to do graph states
    # ghz_dm = DensityMatrix(ghz_st)

    # if density_matrix:
    #     return ghz_dm
    # else:
    #     return ghz_st

def w_state(n: int, verbose=False):
    """Generate a W state.

    Args:
    n (int): Number of qubits.
    verbose (bool, optional): If True, print the generated state. Default is False.

    Returns:
    Statevector: A W state.
    """

    w_str = '1' + '0'*(n-1)
    # give a list of strings of the form '1000', '0100', '0010', '0001'
    w_str_list = [w_str[-i:] + w_str[:-i] for i in range(n)]
    if verbose: print(w_str_list)
    w_st = sum([Statevector.from_label(w).data for w in w_str_list])/np.sqrt(n)
    w_st = Statevector(w_st)
    if verbose: print(w_st)
    return w_st


def random_states(n: int, m: int, is_rho=False, verbose=False):
    """Generate a list of random quantum states or density matrices.

    Args:
    n (int): Number of qubits.
    m (int): Number of random states or density matrices to generate.
    is_rho (bool, optional): If True, generate density matrices. If False, generate state vectors. Default is False.
    verbose (bool, optional): If True, print the generated states or density matrices. Default is False.

    Returns:
    list: A list of random state vectors or density matrices, depending on the value of is_rho.
    """
    random_st = [random_statevector(2**n) for _ in range(m)]
    if verbose: print(random_st)
    random_dm = [random_density_matrix(2**n) for _ in range(m)]
    if is_rho:
        return random_dm
    else:
        return random_st

def ground_state(hamiltonian, return_all=False, return_val=False, verbose=False):
    """Find the ground state of a given Hamiltonian.

    Args:
    hamiltonian (array): A Hamiltonian matrix.
    return_all (bool, optional): If True, return all eigenvalues and eigenvectors. Default is False.
    verbose (bool, optional): If True, print the eigenvalues and eigenvectors. Default is False.

    Returns:
    Statevector or DensityMatrix: The ground state of the Hamiltonian.
    """
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    # energy, state = scipy.linalg.eigh(sum(H_list))
    # # find the idx of the smallest energy
    # idx = np.argmin(energy)
    if verbose: print('eigvals: ', eigvals)
    if return_all:
        return eigvals, eigvecs
    else:
        if return_val:
            return eigvals[0], eigvecs[:, 0]
        else:
            return Statevector(eigvecs[:, 0])
