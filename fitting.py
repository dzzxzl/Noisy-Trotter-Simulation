import numpy as np

fitwithslope = True

class ReadVal:
    '''Data and fitted facs organized in dictionary.

    Attr:

        data (dict): Dictionary of data.

    ''' 

    data: dict
    '''Dictionary of data.
        {
            'data': {'phy_err': [], 'alg_err': [], 'tr_dist': [], 'rel_ent_loc': [], 'rel_ent_glob': [], 'ent_glob_loc_ratio': [], 'ent_dist': []},
            
            'fitted_facs': {'C': C, 'B': B, 'c': c, 'b': b, 'Cp': Cp, 'cp': cp, 'bp': bp}
        }
        '''
    
    def __init__(self, filename):
        '''Read the data and fitted facs from file.
        
        Args:
        filename (str): filename should end with '.npy'. 

        Read data
        {
            'data': {'phy_err': [], 'alg_err': [], 'tr_dist': [], 'rel_ent_loc': [], 'rel_ent_glob': [], 'ent_glob_loc_ratio': [], 'ent_dist': []},
            
            'fitted_facs': {'C': C, 'B': B, 'c': c, 'b': b, 'Cp': Cp, 'cp': cp, 'bp': bp}
        }
        
        '''
        self.data = np.load(filename, allow_pickle=True).item()

class PolyFitting:
    '''Organize fitted values for different r, p, t. The factors are fitted taking n as x-coordinates.
    The class contains a dictionary of C, B, c, b, Cp, cp, bp values each, and contains initial data of different n in the data attribute. The data of n is indexed by n. The class fits the factors over n and keep the values(including slope and intercept) in fitted_facs attribute.
    
    Attr:
        
        r (int): Number of steps.
        t (float): Time.
        p (float): Noise parameter.
        n_list (list): Sorted list of n values.
        data (dict): Dictionary with n as keys and a list of fitted values as values.
        fitted_facs (dict): Fitted factors with respect to n. Structured as
            {'C_s': C_s, 'C_i': C_i, 'B_s': B_s, 'B_i': B_i, 'c_s': c_s, 'c_i': c_i, 'b_s': b_s, 'b_i': b_i}
        C_dict (dict): Dictionary of C values indexed by n.
        B_dict (dict): Dictionary of B values indexed by n.
        c_dict (dict): Dictionary of c values indexed by n.
        b_dict (dict): Dictionary of b values indexed by n.
        Cp_dict (dict): Dictionary of Cp values indexed by n.
        cp_dict (dict): Dictionary of cp values indexed by n.
        bp_dict (dict): Dictionary of bp values indexed by n.

    Methods:

        save(): Save the data and fitted factors to file.
    
    '''
    def __init__(self, r, t, p, n_files):
        '''Initialize the data and fitted factors.
        
        Args:
        r (int): Number of steps.
        t (float): Time.
        p (float): Noise parameter.
        n_files (dict): Dict of filedata indexed by n.
        
        '''
        self.r = r
        self.t = t
        self.p = p
        self.data = {}
        self.fitted_facs = {}
        for n, filedata in n_files.items():
        #     self.data[n] = np.load(filename, allow_pickle=True).item()
            self.data[n] = filedata
        n_list = list(n_files.keys())
        n_list = sorted(n_list)
        self.n_list = n_list
        # global fitwithslope
        # if not fitwithslope:
        C_list = [self.data[n]['fitted_facs']['C'] for n in n_list]
        B_list = [self.data[n]['fitted_facs']['B'] for n in n_list]
        c_list = [self.data[n]['fitted_facs']['c'] for n in n_list]
        b_list = [self.data[n]['fitted_facs']['b'] for n in n_list]
        self.C_dict = {n: self.data[n]['fitted_facs']['C'] for n in n_list}
        self.B_dict = {n: self.data[n]['fitted_facs']['B'] for n in n_list}
        self.c_dict = {n: self.data[n]['fitted_facs']['c'] for n in n_list}
        self.b_dict = {n: self.data[n]['fitted_facs']['b'] for n in n_list}
        self.Cp_dict = {n: self.data[n]['fitted_facs']['Cp'] for n in n_list}
        self.cp_dict = {n: self.data[n]['fitted_facs']['cp'] for n in n_list}
        self.bp_dict = {n: self.data[n]['fitted_facs']['bp'] for n in n_list}
        C_s, C_i = np.polyfit(n_list, C_list, 1)
        B_s, B_i = np.polyfit(n_list, B_list, 1)
        c_s, c_i = np.polyfit(n_list, c_list, 1)
        b_s, b_i = np.polyfit(n_list, b_list, 1)
        self.fitted_facs = {'C_s': C_s, 'C_i': C_i, 'B_s': B_s, 'B_i': B_i, 'c_s': c_s, 'c_i': c_i, 'b_s': b_s, 'b_i': b_i}
            
    def save(self, filename, savetxt=False):
        '''Save the data and fitted facs to file.
        
        Args:
            
            filename (str): filename should end with '.npy'.
            savetxt (boolean): True if also save a txt copy.

        Saved dictionary structure:

        {
            'data': {'C': [], 'B': [], 'c': [], 'b': []},

            'fitted_facs': {'C_s': C_s, 'C_i': C_i, 'B_s': B_s, 'B_i': B_i, 'c_s': c_s, 'c_i': c_i, 'b_s': b_s, 'b_i': b_i}
        }    
            
        '''
        data = {
            'data': {
                'C': [self.data[n]['fitted_facs']['C'] for n in self.n_list],
                'B': [self.data[n]['fitted_facs']['B'] for n in self.n_list],
                'c': [self.data[n]['fitted_facs']['c'] for n in self.n_list],
                'b': [self.data[n]['fitted_facs']['b'] for n in self.n_list]
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
        
class AvgPolyFitting:
    '''This class averages the PolyFitting over all input p.

    Attr:

        polyfits (list): List of PolyFitting instances, indexed by p.
        r (int): Number of steps.
        t (float): Time.
        p_list (list): List of p values.
        n_list (list): Sorted list of n values.
        fitted_facs (dict): Fitted factors with respect to n. Structured as
            {'C_s': C_s, 'C_i': C_i, 'B_s': B_s, 'B_i': B_i, 'c_s': c_s, 'c_i': c_i, 'b_s': b_s, 'b_i': b_i}
        C_dict (dict): Dict of C values.
        B_dict (dict): Dict of B values.
        c_dict (dict): Dict of c values.
        b_dict (dict): Dict of b values.
    
    '''

    def __init__(self, polyfits):
        '''Average the PolyFitting over all input p.
        
        Args:
        polyfits (list): List of PolyFitting instances.
        
        '''
        self.polyfits = {}
        if type(polyfits) == dict:
            polyfits = list(polyfits.values())
        for polyfit in polyfits:
            self.polyfits[polyfit.p] = polyfit
        self.r = polyfits[0].r
        self.t = polyfits[0].t
        self.p_list = [polyfit.p for polyfit in polyfits]
        self.n_list = polyfits[0].n_list
        self.fitted_facs = {}
        self.C_dict = {}
        self.B_dict = {}
        self.c_dict = {}
        self.b_dict = {}
        # global fitwithslope
        # if not fitwithslope:
        for n in self.n_list:
            self.C_dict[n] = np.mean([polyfit.C_dict[n] for polyfit in polyfits])
            self.B_dict[n] = np.mean([polyfit.B_dict[n] for polyfit in polyfits])
            self.c_dict[n] = np.mean([polyfit.c_dict[n] for polyfit in polyfits])
            self.b_dict[n] = np.mean([polyfit.b_dict[n] for polyfit in polyfits])
        C_list = [self.C_dict[n] for n in self.n_list]
        B_list = [self.B_dict[n] for n in self.n_list]
        c_list = [self.c_dict[n] for n in self.n_list]
        b_list = [self.b_dict[n] for n in self.n_list]
        C_s, C_i = np.polyfit(self.n_list, C_list, 1)
        B_s, B_i = np.polyfit(self.n_list, B_list, 1)
        c_s, c_i = np.polyfit(self.n_list, c_list, 1)
        b_s, b_i = np.polyfit(self.n_list, b_list, 1)
        self.fitted_facs = {'C_s': C_s, 'C_i': C_i, 'B_s': B_s, 'B_i': B_i, 'c_s': c_s, 'c_i': c_i, 'b_s': b_s, 'b_i': b_i}
        # else:
        #     for n in self.n_list:
        #         C_dict

class PolyFitting_2D:
    '''This class is the improved version of PolyFitting that also fits over p for the slope and intercept.
    
    Attr:

        r (int): Number of steps.
        t (float): Time.
        p_list (list): List of noise parameters.
        n_list (list): Sorted list of n values.
        data (dict): Dictionary with (n, p) as keys and a list of fitted values as values.
        fitted_facs (dict): Fitted factors with respect to n. Structured as
            {'C_s': C_s, 'C_i': C_i, 'B_s': B_s, 'B_i': B_i, 'c_s': c_s, 'c_i': c_i, 'b_s': b_s, 'b_i': b_i}
        Cp_dict (dict): Dictionary of Cp values indexed by (n, p).
        cp_dict (dict): Dictionary of cp values indexed by (n, p).
        B_dict (dict): Dictionary of B values indexed by (n, p).
        bp_dict (dict): Dictionary of bp values indexed by (n, p).
    
    Methods:

        save(): Save the data and fitted factors to file.
    
    '''
    def __init__(self, r, t, np_files):
        '''Initialize the data and fitted factors.
        
        Args:
            r (int): Number of steps.
            t (float): Time.
            np_files (dict): Dict of filedata indexed by (n, p).
        '''
        self.r = r
        self.t = t
        self.data = {}
        self.fitted_facs = {}
        for (n, p), filedata in np_files.items():
            self.data[(n, p)] = filedata
        n_list = list(set([n for n, p in np_files.keys()]))
        n_list = sorted(n_list)
        self.n_list = n_list
        p_list = list(set([p for n, p in np_files.keys()]))
        p_list = sorted(p_list)
        self.p_list = p_list
        self.Cp_dict = {}
        self.cp_dict = {}
        self.B_dict = {}
        self.bp_dict = {}
        for n in n_list:
            Cp_list = [self.data[(n, p)]['fitted_facs']['Cp'] for p in p_list]
            cp_list = [self.data[(n, p)]['fitted_facs']['cp'] for p in p_list]
            B_list = [self.data[(n, p)]['fitted_facs']['B'] for p in p_list]
            bp_list = [self.data[(n, p)]['fitted_facs']['bp'] for p in p_list]
            
        
class ReadFit:
    '''Read the data and fitted factors from file.
    
    Attr:
    
        data (dict): Dictionary of data. Structured as
        
        {
            'data': {'C': [], 'B': [], 'c': [], 'b': []},

            'fitted_facs': {'C_s': C_s, 'C_i': C_i, 'B_s': B_s, 'B_i': B_i, 'c_s': c_s, 'c_i': c_i, 'b_s': b_s, 'b_i': b_i}
        }   
    
    '''

    def __init__(self, filename):
        '''Read the data and fitted facs from file.
        
        Args:
        filename (str): filename should end with '.npy'.
        
        '''
        self.data = np.load(filename, allow_pickle=True).item()