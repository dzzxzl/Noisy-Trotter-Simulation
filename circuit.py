from qiskit.quantum_info import DensityMatrix, SparsePauliOp
from datetime import datetime
import numpy as np
from qiskit.circuit.library import PermutationGate
# from organized.trotter_error import gen_perm_mat

# The qubits are arranged in the qiskit convention.

def get_datetime():
    '''Return the current date and time in the format "Y%YM%mD%d_h%Hm%Ms%S"'''
    return datetime.now().strftime("Y%YM%mD%d_h%Hm%Ms%S")

def apply_unitary_layer(rho, U):
    '''Apply a unitary layer U to the state rho.
    Return the resulting state.'''
    return U @ rho @ U.conj().T

def apply_single_depolar_noise(rho, p, offset):
    '''Apply a single-qubit depolarizing noise to the offset qubit of the state rho.
    Return the resulting state.'''
    dim = rho.shape[0]
    n = int(np.log2(dim))
    noise_X = SparsePauliOp( 'I'*(n-offset-1) + 'X' + 'I'*(offset), 1 ).to_matrix()
    noise_Y = SparsePauliOp( 'I'*(n-offset-1) + 'Y' + 'I'*(offset), 1 ).to_matrix()
    noise_Z = SparsePauliOp( 'I'*(n-offset-1) + 'Z' + 'I'*(offset), 1 ).to_matrix()
    return (1-p)*rho + p/3*( apply_unitary_layer(rho, noise_X) + apply_unitary_layer(rho, noise_Y) + apply_unitary_layer(rho, noise_Z) )

def apply_single_dephase_noise(rho, p, offset):
    '''Apply a single-qubit depolarizing noise to the offset qubit of the state rho.
    Return the resulting state.'''
    dim = rho.shape[0]
    n = int(np.log2(dim))
    noise_Z = SparsePauliOp( 'I'*(n-offset-1) + 'Z' + 'I'*(offset), 1 ).to_matrix()
    return (1-p)*rho + p*apply_unitary_layer(rho, noise_Z)

def apply_single_ampdamp_noise(rho, p, offset):
    '''Apply a single-qubit amplitude damping noise to the offset qubit of the state rho.
    Return the resulting state.'''
    dim = rho.shape[0]
    n = int(np.log2(dim))
    E0, E1 = np.array([[1, 0], [0, np.sqrt(1-p)]]), np.array([[0, np.sqrt(p)], [0, 0]])
    E0_n, E1_n= np.kron(np.eye(2**(n-offset-1)), np.kron(E0, np.eye(2**offset))), np.kron(np.eye(2**(n-offset-1)), np.kron(E1, np.eye(2**offset)))
    return E0_n @ rho @ E0_n.conj().T + E1_n @ rho @ E1_n.conj().T

def apply_noise_depolar_layer(rho, p):
    '''Apply a layer of single-qubit depolarizing noise to the state rho.
    Return the resulting state.'''
    dim = rho.shape[0]
    n = int(np.log2(dim))
    rho_ret = rho.copy()
    for i in range(n):
        rho_ret = apply_single_depolar_noise(rho_ret, p, i)
    return rho_ret

def apply_noise_layer(rho, p, noise_type='local_depolar'):
    '''Apply a layer of (different types of) noise to the state rho.
    Return the resulting state.'''
    dim = rho.shape[0]
    n = int(np.log2(dim))
    rho_ret = rho.copy()
    if noise_type == 'global_depolar':
        rho_ret = (1-p)*rho_ret + p*np.eye(dim)  # 4/3*p*I ??
    elif noise_type == 'local_depolar':
        for i in range(n):
            rho_ret = apply_single_depolar_noise(rho_ret, p, i)
    elif noise_type == 'local_dephase':
        for i in range(n):
            rho_ret = apply_single_dephase_noise(rho_ret, p, i)
    elif noise_type == 'local_ampdamp':
        for i in range(n):
            rho_ret = apply_single_ampdamp_noise(rho_ret, p, i)
    else:
        raise ValueError('Noise type is not defined')
        
    return rho_ret

def permute_density_matrix(rho, perm):
    '''Permute the density matrix rho according to the permutation perm.
    
    Args:
    rho (np.array): Density matrix to permute.
    perm (list): Permutation to apply.
        perm[j] = i means that the i-th qubit is mapped to the j-th qubit.
    
    Returns:
    np.array: Permuted density matrix.
    '''
    rho_ = rho
    if rho.__class__.__name__ == 'DensityMatrix':
        rho_ = rho.data
    perm_gate = PermutationGate(perm)
    perm_mat = perm_gate.to_matrix()
    # perm_mat = 
    return perm_mat @ rho_ @ perm_mat.conj().T