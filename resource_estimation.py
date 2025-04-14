
class Empirical_Error_Formula:
    def __init__(self, coeffs):
    # def __init__(self, coeffs, n: int, p: int, gamma: float, r: int, t: float):
        self.coeffs = coeffs
        self.C_s, self.C_i, self.c_s, self.c_i, self.B_s, self.B_i, self.b_s, self.b_i = coeffs.values()
        # self.n, self.p, self.gamma, self.r, self.t = n, p, gamma, r, t

    # def one_step_tot_err(self, d: int):
    def accumulated_tot_err(self, n: int, p: int, gamma: float, t: float, r: int, verbose=False):
        """
        Args:
            n (int): Number of qubits.
            p (int): Trotter order.
            gamma (float): noise rate per gate.
            d (int): circuit depth (the index of the Trotter step).
            r (int): Trotter number.
            t (float): evolution time.

        Returns:
            float: The total error in the d-th Trotter step with r Trotter number.
        """
        phy_err_1st_step = (self.C_s*n+self.C_i)*gamma
        self.phy_err_1st_step = phy_err_1st_step
        phy_err_exp_decay = [np.exp(-(self.c_s*n+self.c_i)*d*gamma) for d in range(r)]
        # if verbose: print(phy_err_1st_step, phy_err_exp_decay)
        phy_err_every_step = phy_err_1st_step*np.array(phy_err_exp_decay)

        alg_err_1st_step = (self.B_s*n+self.B_i)*(t**(p+1)/r**(p+1))
        self.alg_err_1st_step = alg_err_1st_step
        alg_err_exp_decay = [np.exp(-(self.b_s*n+self.b_i)*d*gamma) for d in range(r)]
        # if verbose: print(alg_err_1st_step/(t**(p+1)/r**(p+1)), alg_err_exp_decay)
        alg_err_every_step = alg_err_1st_step*np.array(alg_err_exp_decay)

        emp_acc_tot_err = sum(phy_err_every_step) + sum(alg_err_every_step)
        self.emp_acc_tot_err = emp_acc_tot_err
        return min(emp_acc_tot_err, 2)
    # def emp_accumulated_err(self, n: int, p: int, gamma: float, r: int, t: float):
        # return sum([self.emp_step_tot_err(n, p, gamma, d, r, t) for d in range(r)])

def worst_case_err(n: int, t: float, p: int, comm_norm: float, gamma: float, r: int):
    worst_phy_err = comm_norm*t**(p+1)/r**(p)
    # worst_phy_err = tight_bound(h_list, p, t, r)
    worst_alg_err = 2*gamma*n*r
    worst_acc_tot_err = worst_phy_err + worst_alg_err
    return min(worst_acc_tot_err, 2)

def binary_search_opt_r(f, left, right, tol=2, max_iter=1000, verbose=False):
    """
    Minimize a convex function f(r) using binary search, where r is a positive integer.
    
    Parameters:
    f: The convex function to minimize
    left: The starting left bound for binary search
    right: The starting right bound for binary search
    tol: The tolerance for stopping criterion (default is 1)
    max_iter: Maximum number of iterations to avoid infinite loops
    
    Returns:
    r_star: The value of r that minimizes f(r)
    """
    print('-------binary_search_opt_r-------')
    
    # Check if the function at left is decreasing or not
    if f(left) < f(left + 1):  # If f(left) is not decreasing, reduce left
        left //= 10
        print('shrink left to ', left)
        if left < 1:  # Ensure that we don't go below 1
            left = 1
    
    # Check if the function at right is increasing or not
    while f(right) > f(right + 1):  # If f(right) is not increasing, reduce right
        right *= 10
        print('enlarge right to ', right)

    # Perform binary search
    iter_count = 0
    while right - left > tol and iter_count < max_iter:
        if verbose: print(left, right)
        mid1 = left + (right - left) // 3  # Use integer division for integer values of r
        mid2 = right - (right - left) // 3
        
        if f(mid1) < f(mid2):
            right = mid2
        else:
            left = mid1
        
        iter_count += 1

    if iter_count == max_iter:
        print('Maximum number of iterations reached. Returning the closest integer to the minimum value.')
    
    # After the loop ends, return the closest integer to the minimum value
    r_star = (left + right) // 2  # Return as an integer
    return r_star

import numpy as np
from quantum_simulation_recipe.measure import commutator, norm
from qiskit.quantum_info import SparsePauliOp

def analytic_loose_commutator_bound(n, J, h, dt, pbc=False, verbose=False):
    if pbc:
        c1 = 16*J**2*h*(n) + 4*J**2*h*(n)
        c2 = 8*(n)*J**2*h
    else:
        # c1 = 16*J**2*h*(n-1) + 4*J**2*h*(n-2)
        # c2 = 8*(n-1)*J**2*h
        if n % 2 == 1:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
        else:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)


    if verbose: print(f'c1 (analy)={c1}, c2={c2}')
    analytic_error_bound = c1 * dt**3 / 12 + c2 * dt**3 / 24
    return analytic_error_bound

def tight_bound(h_list: list, order: int, t: float, r: int, type='spectral', verbose=False):
    L = len(h_list)
    if isinstance(h_list[0], np.ndarray):
        d = h_list[0].shape[0]
    elif isinstance(h_list[0], SparsePauliOp):
        n = h_list[0].num_qubits
        d = 2**n
    # elif isinstance(h_list[0], csr_matrix):
    #     d = h_list[0].todense().shape[0]
    else:
        raise ValueError('Hamiltonian type is not defined')

    if order == 1:
        a_comm = 0
        for i in range(0, L-1):
            # if isinstance(h_list[i], np.ndarray):
            #     temp = np.zeros((d, d), dtype=complex)
            # else:
            #     temp = SparsePauliOp.from_list([("I"*n, 0)])
            
            # for j in range(i + 1, L):
            #     temp += commutator(h_list[i], h_list[j])
            temp = sum([commutator(h_list[i], h_list[j]) for j in range(i + 1, L)])
            a_comm += norm(temp, ord=type)

        if type == 'spectral':
            error = a_comm * t**2 / (2*r)
        elif type == 'fro':
            error = a_comm * t**2 / (2*r*np.sqrt(d))
        else:
            raise ValueError(f'type={type} is not defined')
    elif order == 2:
        c1 = 0
        c2 = 0
        for i in range(0, L-1):
            # if isinstance(h_list[i], np.ndarray):
            #     temp = np.zeros((d, d), dtype=complex)
            # else:
            #     temp = SparsePauliOp.from_list([("I"*n, 0)])
            # for j in range(i + 1, L):
            #     temp += h_list[j]
            temp = sum(h_list[i+1:])
            # h_sum3 = sum(h[k] for k in range(i+1, L))
            # print(h_sum3.shape)
            # h_sum2 = sum(h[k] for k in range(i+1, L))
            c1 += norm(commutator(temp, commutator(temp, h_list[i])), ord=type) 
            # c1 = norm(commutator(h[0]+h[1], commutator(h[1]+h[2], h[0]))) + norm(commutator(h[2], commutator(h[2], h[1])))
            # c2 = norm(commutator(h[0], commutator(h[0],h[1]+h[2]))) + norm(commutator(h[1], commutator(h[1], h[2])))
            c2 += norm(commutator(h_list[i], commutator(h_list[i], temp)), ord=type)
        if type == 'spectral':
            error = c1 * t**3 / r**2 / 12 + c2 *  t**3 / r**2 / 24 
        elif type == 'fro':
            # print(c1, c2)
            error = c1 * t**3 / r**2 / 12 / np.sqrt(d) + c2 *  t**3 / r**2 / 24 / np.sqrt(d)
            # print('random input:', error)
        elif type == '4':
            error = c1 * t**3 / r**2 / 12 / d**(1/4) + c2 *  t**3 / r**2 / 24 / d**(1/4)
        else:
            raise ValueError(f'type={type} is not defined')
    else: 
        raise ValueError(f'higer order (order={order}) is not defined')

    if verbose: print(f'c1={c1}, c2={c2}')

    return error