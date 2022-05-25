# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim as egm
from numba import njit, int64, double, boolean, int32,void

class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par

        par.T = 70

        # Model parameters
        par.rho = 0.5
        par.beta = 0.96
        par.alpha = 0.75
        par. R = 1.04
        par.W = 1
        par.sigma_xi = 0.05
        par.sigma_eta = 0.1

        # Grids and numerical integration
        par.m_max = 50
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 50
        par.a_phi = 1.1  # Curvature parameters
        par.h_max = 1.0
        par.p_phi = 1.0 # Curvature parameters

        par.Nxi = 8
        par.Nm = 100
        par.Na = 100
        par.Nh = 100

        par.NTh = 7 # number of possible exercise bundles 

        par.kappa = 0.05   # abilty to transfer exercise into health
        par.gamma = 0.05  # health decay

        par.Nm_b = 50

        par.Rh = 1

         # 6. simulation
        par.sim_mini = 2.5 # initial m in simulation
        par.simN = 100 # number of persons in simulation
        par.simT = par.T# number of periods in simulation
        
    def create_hour_bunches(self):
        #exercise = np.linspace(0, 3, num=50)* 7 * 52
        exercise =np.array([0, 1, 2, 3]) * 7 * 52 # yearly exercise              
        work = np.array([0, 1000, 2000, 2250, 2500, 3000])         # from Iskhakov and Keane (2021)
        #work = np.linspace(0, 3000, num=50)
        hour_boundles = []                                         # Data container for work and exercise
        for e_h in exercise:
            for w_h in work:
                hour_boundles.append([e_h, w_h])                       # append(work and exercise boundle
        hour_boundles = np.array([[0,0], [550,2000], [0,2000], [1000,2500], [550,2500], [550,3000], [0,3000]])        
        return np.array(hour_boundles)
    
    def create_grids(self):
        par = self.par

        # Check parameters
        assert (par.rho >= 0), 'not rho > 0'

        # Shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
        #print(par.xi)

        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)]) 

        # Health states
        par.grid_h = tools.nonlinspace(0+1e-4,par.h_max,par.Nh,par.p_phi)
        par.T_boundles = self.create_hour_bunches()
        par.NT = par.T_boundles.shape[0]

        par.Nshocks = par.xi * par.Na
        # Set seed
        np.random.seed(2020)

    def solve(self, print_iteration_number = True):
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T, par.NT, par.Nm, par.Nh)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        
        
        # Last period, (= consume all) 
        for i_h in range(par.Nh):
            for i_t, T_plus in enumerate(par.T_boundles):
                sol.m[par.T-1,i_t,:,i_h] = par.grid_m
                sol.c[par.T-1,i_t,:,i_h] = par.grid_m
                # h= 0 = dead
                sol.v[par.T-1,i_t,:,i_h] = egm.util(sol.c[par.T-1,i_t,:,i_h],T_plus,par.T-1,par, h = 0)
        
        # Before last period
        # T_plus is time choice [T^w, T^H], e.g. [5, 10]
        for t in range(par.T-2,-1,-1):
            if print_iteration_number:
                print(f'Evaluating period: {t}')
            #Choice specific function
            for i_h, p in enumerate(par.grid_h):
                #if p < 0.001:
                #    for i_t, T_plus in enumerate(par.T_boundles):
                #        T_plus = [0,0]
                #        # Solve model with EGM
                #        c,v = egm.EGM(sol,T_plus,p,t,par)
                #        sol.c[t,i_t,:,i_h] = c
                #        sol.v[t,i_t,:,i_h] = v
                #        continue
                #else:
                for i_t, T_plus in enumerate(par.T_boundles):

                    # Solve model with EGM
                    c,v = egm.EGM(sol,T_plus,p,t,par)

                    sol.c[t,i_t,:,i_h] = c
                    sol.v[t,i_t,:,i_h] = v