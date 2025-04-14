from metric import state_alg_err, state_phy_err, ob_alg_err, ob_phy_err, tr_dist_approx, rel_ent_loc_avg, rel_ent_glob, rel_ent_approx
from circuit import apply_noise_depolar_layer, apply_noise_layer
from qiskit.quantum_info import DensityMatrix
import numpy as np
from tqdm import tqdm
from quantum_simulation_recipe.trotter import pf 


class Evolve_state:
    '''Evolve the state rho under the unitary U_dt for r steps.

    Attr:

        rho_init (np.array): Initial state.
        rho (np.array): Evolved state.
        alg_err_list (list): List of algorithmic errors at each step.
        phy_err_list (list): List of physical errors at each step.
        tr_dist_list (list): List of trace distances at each step.
        rel_ent_loc_list (list): List of relative entropies of the reduced density matrices at each step.
        rel_ent_glob_list (list): List of relative entropies of the state and the maximally mixed state at each step.
        ent_glob_loc_ratio_list (list): List of the ratio of the relative entropy of the state and the maximally mixed state to the relative entropy of the reduced density matrices at each step.
        ent_dist_list (list): List of entropy distances at each step.
        fitted_facs (dict): Dictionary of fitted factors for the algorithmic error. The fitted facs include {'C', 'B', 'c', 'b', 'Cp', 'cp', 'bp'}
    
    Methods:

        save(): Save the data and fitted facs to file.

    '''

    def __init__(self, U_dt, U_dt_exact, r, rho, gamma, p, t, noise_type='local_depolar', verbose=False, detail=False):
    # def __init__(self, U_dt, U_dt_exact, r, rho, p, dt_p, noise_type='local_depolar', verbose=False):
        '''Evolve the state rho under the unitary U_dt for r steps.
        
        Args:
            U_dt (np.array): Unitary to evolve the state.
            U_dt_exact (np.array): Exact unitary to compare the state evolution.
            r (int): Number of steps to evolve the state.
            rho (np.array): Initial state to evolve.
            gamma (float): Noise strength.
            p (int): order of Trotter formula.
            # dt_p (float): dt^{Trotter order + 1}. Magnitude of the algorithmic error.

        Computed attrs:
            rho_init (np.array): Initial state.
            rho (np.array): Evolved state.
            alg_err_list (list): List of algorithmic errors at each step.
            phy_err_list (list): List of physical errors at each step.
            tr_dist_list (list): List of trace distances at each step.
            rel_ent_loc_list (list): List of relative entropies of the reduced density matrices at each step.
            rel_ent_glob_list (list): List of relative entropies of the state and the maximally mixed state at each step.
            ent_dist_list (list): List of entropy distances at each step.
            fitted_facs (dict): Dictionary of fitted factors for the algorithmic error. The fitted facs include {'C', 'B', 'c', 'b', 'Cp', 'cp', 'bp'}
    
        '''
        if rho.__class__.__name__ == 'DensityMatrix':
            rho = rho.data
        elif rho.__class__.__name__ == 'Statevector':
            rho = DensityMatrix(rho).data
        self.rho_init = rho
        self.U_dt = U_dt
        self.U_dt_exact = U_dt_exact
        self.r = r
        self.rho = rho.copy()
        self.gamma = gamma
        self.p = p
        self.t = t
        self.dt_p = (t/r)**(p+1)
        self.alg_err_list = []
        self.phy_err_list = []
        self.tr_dist_list = []
        self.rel_ent_loc_list = []
        self.rel_ent_glob_list = []
        self.ent_glob_loc_ratio_list = []
        self.ent_dist_list = []
        self.fitted_facs = {
            'C': 0, 'B': 0, 'c': 0, 'b': 0, 'Cp': 0, 'cp': 0, 'bp': 0
        }
        self.noise_type = noise_type

        for i in tqdm(range(r)):  # tqdm is progress bar with percentage
        # for i in range(r):
            # progress bar with percentage
            # print(f'\r{100*i/r:.2f}%', end='')
            # progress_bar(i, r)
            if verbose:
                print('r =', i, end=' ')
            alg_err = state_alg_err( self.U_dt, self.U_dt_exact, self.rho)
            self.alg_err_list.append(alg_err)
            self.rho = self.U_dt @ self.rho @ self.U_dt.conj().T

            phy_err = state_phy_err(self.rho, self.gamma)
            self.phy_err_list.append(phy_err)
            tr_dist = tr_dist_approx(self.rho, self.gamma)
            if detail:
                self.tr_dist_list.append(tr_dist)
                rel_ent_loc = rel_ent_loc_avg(self.rho)
                self.rel_ent_loc_list.append(rel_ent_loc)
                rel_ent_glob_ = rel_ent_glob(self.rho)
                self.rel_ent_glob_list.append(rel_ent_glob_)
                self.ent_glob_loc_ratio_list.append(rel_ent_glob_ / rel_ent_loc)
                ent_dist = rel_ent_approx(self.rho, self.gamma)
                self.ent_dist_list.append(ent_dist)
            self.rho = apply_noise_layer(self.rho, gamma, noise_type)
            # self.rho = apply_noise_depolar_layer(self.rho, p)
        if verbose:
            print()
        
        cp, log_Cp = np.polyfit(range(r), np.log(self.phy_err_list), 1)
        bp, log_Bdt_p = np.polyfit(range(r), np.log(self.alg_err_list), 1)

        Cp = np.exp(log_Cp)
        cp = -cp 
        B = np.exp(log_Bdt_p) / self.dt_p
        bp = - bp 

        C = Cp / self.gamma 
        c = cp / self.gamma
        b = bp / self.gamma

        self.fitted_facs = {
            'C': C, 'B': B, 'c': c, 'b': b, 'Cp': Cp, 'cp': cp, 'bp': bp
        }

    def save(self, filename, savetxt=False):
        '''Save the data and fitted facs to file. The data and fitted facs are organized into a dictionary.

        Args:
            filename (str): Name of the file to save the data and fitted facs. filename should end with '.npy'.
            savetxt (boolean): True if also save a txt copy.
        
        Dictionary structure:

        {
            'data': {'phy_err': [], 'alg_err': [], 'tr_dist': [], 'rel_ent_loc': [], 'rel_ent_glob': [], 'ent_glob_loc_ratio': [], 'ent_dist': []},
            
            'fitted_facs': {'C': C, 'B': B, 'c': c, 'b': b, 'Cp': Cp, 'cp': cp, 'bp': bp}
        }

        '''
        data = {
            'data': {
                'phy_err': self.phy_err_list,
                'alg_err': self.alg_err_list,
                'tr_dist': self.tr_dist_list,
                'rel_ent_loc': self.rel_ent_loc_list,
                'rel_ent_glob': self.rel_ent_glob_list,
                'ent_glob_loc_ratio': self.ent_glob_loc_ratio_list,
                'ent_dist': self.ent_dist_list
            },
            'fitted_facs': self.fitted_facs
        }
        if filename.endswith('.npy'):
            np.save(filename, data)
            if savetxt:
                txt_ = filename[:-4] + '.txt'
                with open(txt_, 'w') as f:
                    f.write(str(data))
        else:
            raise ValueError('filename should end with ".npy" or ".txt".')

class Evolve_ob:
    '''Evolve the observable in the Heisenberg picture under U_dt for r steps.
    
    Attr:
    
        ob_init (np.array): Initial observable.
        ob (np.array): Evolved observable.
        alg_err_list (list): List of algorithmic errors at each step.
        phy_err_list (list): List of physical errors at each step.

    Methods:

        save(): Save the data to file.
        
    '''

    def __init__(self, U_dt, U_dt_exact, r, ob, gamma, state=None):
        '''Evolve the observable in the Heisenberg picture under U_dt for r steps.
        
         Args:
            U_dt (np.array): Unitary to evolve the state.
            U_dt_exact (np.array): Exact unitary to compare the state evolution.
            r (int): Number of steps to evolve the state.
            rho (np.array): Initial state to evolve.
            gamma (float): Noise strength.
            state (np.array): State to estimate the observable expecation value error. If None, evolve the observable under the unitary U_dt.
        '''

        self.U_dt = U_dt
        self.U_dt_exact = U_dt_exact
        self.r = r
        self.ob_init = ob
        self.ob = ob.copy()
        self.alg_err_list = []
        self.phy_err_list = []
        self.gamma = gamma
        self.state_alg_err_list = []
        self.state_phy_err_list = []
        self.expval_list = []
        if state is None:
            self.state = np.zeros((ob.shape[0], ob.shape[0]), dtype=complex)
            self.state[0, 0] = 1
        else:
            self.state = state.copy()

        for i in tqdm(range(r)):  # tqdm is progress bar with percentage
            # print(f'progress \r{100*i/r:.2f}%', end='')
            # progress_bar(i, r)
            alg_err = ob_alg_err( self.U_dt, self.U_dt_exact, self.ob)
            state_alg_err =  ob_alg_err( self.U_dt, self.U_dt_exact, self.ob, self.state)
            self.alg_err_list.append(alg_err)
            self.ob = self.U_dt.conj().T @ self.ob @ self.U_dt
            self.state_alg_err_list.append(state_alg_err)
            expval = np.trace(self.state @ self.ob)
            self.expval_list.append(expval)

            phy_err = ob_phy_err( self.ob, self.gamma)
            self.phy_err_list.append(phy_err)
            state_phy_err = ob_phy_err( self.ob, self.gamma, self.state)
            self.state_phy_err_list.append(state_phy_err)
            self.ob = apply_noise_depolar_layer(self.ob, self.gamma)


def progress_bar(current, total, bar_length=50):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    
    print(f'\r[{arrow}{spaces}] {int(percent * 100)}%', end='')


class Evolve_state_random_H:
    def __init__(self, model, h_name, random_par_gen, r, rho, gamma, p, t, noise_type='local_depolar', verbose=False, detail=False, only_ratio=True, seed=42, **base_par):
        # self.rho_init = rho
        if rho.__class__.__name__ == 'DensityMatrix':
            rho = rho.data
        elif rho.__class__.__name__ == 'Statevector':
            rho = DensityMatrix(rho).data
        self.rho = rho.copy()
        self.model = model
        # self.random_par_gen = random_par_gen,
        self.r = r 
        self.gamma = gamma 
        self.p = p
        self.t = t 
        self.noise_type = noise_type
        self.verbose = verbose
        self.dt_p = (t/r)**(p+1)
        self.rel_ent_loc_list = []
        self.rel_ent_glob_list = []
        self.ent_glob_loc_ratio_list = []

        for i in tqdm(range(r)):
            cur_seed = seed + i 
            model_par = random_par_gen(seed=cur_seed)
            cur_model = self.model(**model_par)
            cur_h_list = [term.to_matrix() for term in getattr(cur_model, h_name)]
            U_of_one_layer = pf(cur_h_list, t/r, 1)
            self.rho = U_of_one_layer @ self.rho @ U_of_one_layer.conj().T
            
            rel_ent_loc = rel_ent_loc_avg(self.rho)
            self.rel_ent_loc_list.append(rel_ent_loc)
            rel_ent_glob_ = rel_ent_glob(self.rho)
            self.rel_ent_glob_list.append(rel_ent_glob_)
            self.ent_glob_loc_ratio_list.append(rel_ent_glob_ / rel_ent_loc)

            if not only_ratio:
                pass

            self.rho = apply_noise_layer(self.rho, gamma, noise_type)
            
    def save(self, filename, savetxt=False):
        data = {
            'data': {
                'rel_ent_loc': self.rel_ent_loc_list,
                'rel_ent_glob': self.rel_ent_glob_list,
                'ent_glob_loc_ratio': self.ent_glob_loc_ratio_list
            }
        }
        if filename.endswith('.npy'):
            np.save(filename, data)
            if savetxt:
                txt_ = filename[:-4] + '.txt'
                with open(txt_, 'w') as f:
                    f.write(str(data))
        else:
            raise ValueError('filename should end with ".npy" or ".txt".')

class RandomParGen:
    def __init__(self, n, value_ranges: dict, base_par):
        self.n = n
        self.base_par = base_par
        self.value_ranges = value_ranges

    def __call__(self, seed=42):
        cur_par = self.base_par.copy()
        np.random.seed(seed)
        for key_name in self.value_ranges.keys():
            cur_par[key_name] = np.random.uniform(self.value_ranges[key_name][0], self.value_ranges[key_name][1])
        return cur_par