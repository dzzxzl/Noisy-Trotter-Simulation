import cmath
import numpy as np
import scipy
import scipy.optimize as opt
from qiskit.quantum_info import partial_trace
from circuit import apply_noise_depolar_layer, permute_density_matrix

def schatten_norm(mat, order=1):
    '''
    Args:
        order: 1 or 'nuc'.     
    '''
    if order == 1:
        return np.linalg.norm(mat, ord='nuc')
    elif order == 'inf':
        return np.linalg.norm(mat, ord=2)
    else:
        raise ValueError('Invalid order')

def state_alg_err(U, V, rho):
    '''
    Args:
    U (np.array): Unitary matrix.
    V (np.array): Unitary matrix.

    Return the trace distance(1/2 of the trace norm of the difference of the two states) between U*rho*U^dagger and V*rho*V^dagger.'''
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho.data
    return 0.5*np.linalg.norm( U @ rho @ U.conj().T - V @ rho @ V.conj().T, ord='nuc') # 'nuc' is nuclear norm: the sum of all singular values

def state_phy_err(rho, p):
    '''Return the trace distance between rho and the rho after one layer of depolarizing noise of strength p.'''
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho.data
    return 0.5*np.linalg.norm( rho - apply_noise_depolar_layer(rho, p), ord='nuc')

def ob_alg_err(U, V, O, state=None):
    '''
    Args:
    U (np.array): Unitary matrix.
    V (np.array): Unitary matrix.
    O (np.array): Observable.
    state (np.array): State. 
    
    If state is None:
        Return the operator norm of the difference of the two observables U^dagger*O*U and V^dagger*O*V.
    If state is not None:
        Return the difference between the expectation values of the two observables U^dagger*O*U and V^dagger*O*V with respect to the state.
    '''
    if state is None:
        return np.linalg.norm( U.conj().T @ O @ U - V.conj().T @ O @ V, ord=2) # '2' is operator(spectral) norm: the largest singular value
    else:
        # return 0.5*np.linalg.norm( state @ (U.conj().T @ O @ U - V.conj().T @ O @ V), ord='nuc')
        return abs( np.trace( state @ (U.conj().T @ O @ U - V.conj().T @ O @ V)) )

def ob_phy_err(O, p, state=None):
    '''Return the operator norm of the difference of the two observables O and the observable after one layer of depolarizing noise of strength p.
    If state is not None:
        Return the difference between the expectation values of the two observables N(O) and O with respect to the state.
    '''
    if state is None:
        return np.linalg.norm( O - apply_noise_depolar_layer(O, p), ord=2)
    else:
        return abs( np.trace( state @ ( O - apply_noise_depolar_layer(O, p) )) )

def relative_entropy(rho, sigma):
    '''Return the relative entropy of rho and sigma.'''
    return np.trace( rho @ ( scipy.linalg.logm(rho) - scipy.linalg.logm(sigma) ) )

def entropy_1(rho):
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho
    return -np.trace( rho @ scipy.linalg.logm(rho) )

def rho_loc(rho, idx):
    '''Return the reduced density matrix tracing out the qubit at index idx and replacing with a maximally mixed state. idx can be an integer or a list of integers.'''
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho.data
    if type(idx) == int:
        idx = [idx]
    n = int(np.log2(rho.shape[0]))
    idx = sorted(idx)
    idxc = list(set(range(n)) - set(idx))
    perm = idx + idxc
    perm_inv = [perm.index(i) for i in range(n)]
    reduced_op = partial_trace(rho, idx)
    idx_len = len(idx)
    reduced_op_tensor_id = reduced_op.tensor(0.5**idx_len * np.eye(2**idx_len)).to_operator().to_matrix()
    ret_op = permute_density_matrix(reduced_op_tensor_id, perm_inv)
    return ret_op

def rel_ent_loc(rho, idx):
    '''Return the relative entropy of the reduced density matrix tracing out the qubit at index idx and replacing with a maximally mixed state and the reduced density matrix of the original state.'''
    return relative_entropy(rho, rho_loc(rho, idx))

def rel_ent_loc_1(rho, idx):
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho.data
    if type(idx) == int:
        idx = [idx]
    rho_loc_ = partial_trace(rho, idx)
    # return 1 + entropy_1(rho)
    return np.log(2) * len(idx) + entropy_1(rho_loc_) - entropy_1(rho)

def rel_ent_loc_avg(rho):
    '''Return the average relative entropy of the reduced density matrices tracing out each qubit and replacing with a maximally mixed state.'''
    n = int(np.log2(rho.shape[0]))
    # return sum([rel_ent_loc(rho, i) for i in range(n)])/n
    rho_loc_ = np.sum([rho_loc(rho, i) for i in range(n)], axis=0)
    rho_loc_ = rho_loc_/n
    return relative_entropy(rho, rho_loc_)

def rel_ent_loc_avg_1(rho):
    n = int(np.log2(rho.shape[0]))
    return np.sum([rel_ent_loc_1(rho, i) for i in range(n)])/n

def rel_ent_glob(rho):
    '''Return the relative entropy of the state rho and the maximally mixed state.'''
    return relative_entropy(rho, np.eye(rho.shape[0])/rho.shape[0])

def rel_ent_glob_1(rho):
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho.data
    n = int(np.log2(rho.shape[0]))
    return np.log(2) * n - entropy_1(rho)

def tr_dist_loc(rho, idx):
    '''Return the trace distance of the reduced density matrix tracing out the qubit at index idx and replacing with a maximally mixed state and the reduced density matrix of the original state.'''
    return 0.5*np.linalg.norm( rho - rho_loc(rho, idx), ord='nuc')

def tr_dist_avg(rho):
    '''Return the average trace distance of the reduced density matrices tracing out each qubit and replacing with a maximally mixed state.'''
    n = int(np.log2(rho.shape[0]))
    # return sum([tr_dist_loc(rho, i) for i in range(n)])/n
    rho_loc_ = np.sum([rho_loc(rho, i) for i in range(n)], axis=0)
    rho_loc_ = rho_loc_/n
    return 0.5 * np.linalg.norm( rho - rho_loc_, ord='nuc')

def tr_dist_approx(rho, p):
    '''
    Args:
    rho(DensityMatrix, np.array): state
    p: noise strength

    Return the trace distance approximation of the phy err.'''
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho.data
    n = int(np.log2(rho.shape[0]))
    return n * p * tr_dist_avg(rho)

def rel_ent_approx(rho, p):
    '''
    Args:
    rho(DensityMatrix, np.array): state
    p: noise strength
    
    Return the relative entropy approximation of the phy err.'''
    if rho.__class__.__name__ == 'DensityMatrix':
        rho = rho.data
    n = int(np.log2(rho.shape[0]))
    return 0.5 * n * p * np.sqrt(2 * rel_ent_loc_avg(rho))

# def rho_loc(rho, idx):
#     '''Return the reduced density matrix tracing out the qubit at index idx and replacing with a maximally mixed state. idx can be an integer or a list of integers.'''
#     if rho.__class__.__name__ == 'DensityMatrix':
#         rho = rho.data
#     if type(idx) == int:
#         idx = [idx]
#     n = int(np.log2(rho.shape[0]))
#     idx = sorted(idx)
#     idxc = list(set(range(n)) - set(idx))
#     perm = idx + idxc
#     perm_inv = [perm.index(i) for i in range(n)]
#     rho_ = np.zeros_like(rho)
#     for i in range(2**len(idx)):
#         for j in range(2**len(idxc)):
#             for k in range(2**len(idx)):
#                 for l in range(2**len(idxc)):
#                     rho_[i*2**len(idxc)+j, k*2**len(idxc)+l] = rho[perm_inv[i]*2**len(idxc)+j, perm_inv[k]*2**len(idxc)+l]
#     return np.eye(2**len(idxc))/2**len(idxc) + rho_

class WeakDiamondDistance:
    '''This class computes the state maximizing the algorithmic error between unitary U and V

    Attr:

        U (np.array): Unitary matrix.
        V (np.array): Unitary matrix.
        distance (float): The optimal distance.
        optimal_state (np.array): The optimal state.

    '''
    def __init__(self, U, V, x0=None, maxiter=1000, disp=False, atol=None):
        '''
        Args:
            U (np.array): Unitary matrix.
            V (np.array): Unitary matrix.
            x0 (np.array): Initial guess of the coefficients. The coeffs are the magnitudes and the phases with respect to the eigenbasis of U^daggerV. x0 = [*pi_s, *ai_s]. pi_s is a list of the magnitudes of the coefficients and ai_s is a list of the phases of the coefficients. The length of pi_s and ai_s is the dimension of the unitary matrix. If x0 is None, the initial guess will be assigned value randomly. 
            disp (bool): If True, print the minimization process.
            atol (float): Specify the scale of the optimal distance. If atol is None, it is taken to be the objective value at the initial x0 guess.
        '''
        self.U = U
        self.V = V
        # self.order = order
        self.disp = disp
        self.maxiter = maxiter
        if x0 is None:
            dim = U.shape[0]
            x0 = np.random.rand(dim)
            x0 = x0 / np.linalg.norm(x0)
            x0 = np.concatenate((x0, np.zeros(dim)))
        self.minimize(x0, atol=atol)

    def minimize(self, x0, atol=None):
        uu = self.U.conj().T @ self.V
        eigs_, eigvecs_ = scipy.linalg.schur(uu)

        # eigs_, eigvecs_ = np.linalg.eig(uu)

        eigs = eigs_.diagonal()
        eigvecs = eigvecs_
        # if self.disp:
        #     print('Eigenvalues: ', eigs)
        thetai_s = np.angle(eigs)
        abs_s = np.abs(eigs)
        if atol is None:
            atol = abs(WeakDiamondDistance.obj_func(x0, thetai_s, 1))
        constraints = (
            {'type': 'eq',
            'fun': lambda x: WeakDiamondDistance.normalization_val(x) - 1}
        )
        # print('atol: ', atol)
        options = {'disp': self.disp, 'maxiter': self.maxiter}
        res = opt.minimize(WeakDiamondDistance.obj_func, x0, args=(thetai_s,1/atol), method='SLSQP', constraints=constraints, callback=None, options=options)
        self.x = res.x
        self.distance =  - WeakDiamondDistance.obj_func(self.x, thetai_s, 1)
        if self.disp:
            print('Optimal distance: ', self.distance)
        optimal_statevec = np.zeros((self.U.shape[0],), dtype=complex)
        
        # for i in range(self.U.shape[0]):
            # optimal_statevec += self.x[i] * eigvecs[:,i]
        optimal_coefs = WeakDiamondDistance.extract_coeffs(self.x)
        for i in range(self.U.shape[0]):
            optimal_statevec += optimal_coefs[i] * eigvecs[:,i]
        self.optimal_state = np.outer(optimal_statevec, optimal_statevec.conj())
        # check
        # dim = self.U.shape[0]
        # act_distance = distance_given_state(self.U.conj().T @ self.V, np.eye(dim), self.optimal_state)
        # act_distance_ = distance_given_state(self.U, self.V, self.optimal_state)
        # if self.disp:
        #     print('x: ', self.x)
        #     print('Actual distance: ', act_distance)
        #     print('Actual distance _: ', act_distance_)

    def normalization_val(x):
        dim = len(x) // 2
        pi_s = x[:dim]
        pi_s = np.array(pi_s)
        sq_sum = np.sum( pi_s * pi_s )
        return sq_sum
    
    def extract_coeffs(x):
        x_len = len(x)
        dim = x_len // 2
        pi_s = x[:x_len//2]
        pi_s_normalized = np.array(pi_s)
        ai_s = x[x_len//2:]
        ret = []
        for i in range(dim):
            ret.append(cmath.rect(pi_s_normalized[i], ai_s[i]))
        return ret

    def obj_func(x, *args):
        x_len = len(x)
        dim = x_len // 2
        pi_s = x[:x_len//2]
        pi_s_normalized = np.array(pi_s)
        ai_s = x[x_len//2:]
        thetai_s = args[0]
        zoom = args[1]
        mat = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                mat[i][j] += ( cmath.exp( 1j * (thetai_s[i] - thetai_s[j]) ) - 1 )\
                      * pi_s_normalized[i] * pi_s_normalized[j]  * cmath.exp( 1j * (ai_s[i] - ai_s[j]) )
        return (- 0.5 * schatten_norm(mat, order=1))*zoom


def worst_state(U_dt_exact, U_dt_appro, verbose=False):
    U_diff = U_dt_exact - U_dt_appro
    if verbose: print(np.linalg.norm(U_diff, ord=2))

    # # s, U = np.linalg.eigh(U_dt - U_dt_exact)
    # s, U, ss = np.linalg.svd(U_dt - U_dt_exact)
    # # if abs(s[-1]) > abs(s[0]):
    # #     print("-1")
    # #     rho_init = np.outer(U[:,-1], U[:,-1].conj())
    # # else:
    # #     print("1")
    # #     rho_init = np.outer(U[:,0], U[:,0].conj())
    # v0 = ss[0].conj().T
    # # print(v0.shape)
    # rho_init = np.outer(v0, v0.conj())
    # # rho_init = np.outer(v0.T, v0.T.conj())
    # # print(rho_init.shape)
    # # for elem in s:
    # #     print(elem, end=' ')
    # # print()
    wdd = WeakDiamondDistance(U_dt_appro, U_dt_exact, disp=True, maxiter=1000)
    rho_init = wdd.optimal_state

    return rho_init

def worst_holistic(U_exact, U_appro):
    # U_holistic = pf(H_list, t, r)
    # print('is all close? ', np.allclose(U_holistic, np.linalg.matrix_power(U_appro, r) ) )
    # U_holistic_exact = expH(sum(H_list), t)
    # wdd = WeakDiamondDistance(U_holistic, U_holistic_exact, disp=True, maxiter=1000)
    wdd = WeakDiamondDistance(U_appro, U_exact, disp=True, maxiter=1000)
    # print(wdd.distance)
    rho_init = wdd.optimal_state

    return rho_init

def worst_state_svd(U_dt_exact, U_dt_appro, return_vec=True, verbose=False):
    U_diff = U_dt_exact - U_dt_appro

    U, S, V = np.linalg.svd(U_diff)
    v0 = V[0].conj().T
    if verbose: 
        print()
        print(f'achieve spectral norm {np.linalg.norm(U_diff, ord=2)} (square root of largest eigenval): {np.linalg.norm(U_diff@v0)}')

    if return_vec:
        return v0
    else:
        rho_init = np.outer(v0, v0.conj())
        return rho_init