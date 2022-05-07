# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim as egm


class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par

        par.T = 10

        # Model parameters
        par.rho = 2
        par.beta = 0.96
        par.alpha = 0.75
        par.kappa = 0.5
        par. R = 1.04
        par.W = 1
        par.sigma_xi = 0.05
        par.sigma_eta = 0.1

        # Grids and numerical integration
        par.m_max = 10
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 10
        par.a_phi = 1.1  # Curvature parameters
        par.p_max = 2.0
        par.p_phi = 1.0 # Curvature parameters

        par.Nxi = 8
        par.Nm = 150
        par.Na = 150
        par.Nh = 100

        par.NTh = 7 # number of possible exercise bundles 

        par.kappa = 1   # abilty to transfer exercise into health
        par.gamma = 0.95  # health decay

        par.Nm_b = 50
        
    def create_hour_bunches(self):
        exercise = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3]) * 50 # yearly exercise              
        work = np.array([0, 1000, 2000, 2250, 2500, 3000])               # from Iskhakov and Keane (2021)
        hour_boundles = []
        for e_h in exercise:
            for w_h in work:
                hour_boundles.append([e_h, w_h])
        return np.array(hour_boundles)
    
    def create_grids(self):
        par = self.par

        # Check parameters
        assert (par.rho >= 0), 'not rho > 0'

        # Shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
        print(par.xi)

        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)]) 

        # Health states
        par.grid_h = tools.nonlinspace(0+1e-4,par.p_max,par.Nh,par.p_phi)
        par.T_boundles = self.create_hour_bunches()
        par.NT = par.T_boundles.shape[0]

        # Set seed
        np.random.seed(2020)

    def solve(self):
        print("??")
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T, par.NT, par.Nm, par.Nh)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        # Last period, (= consume all) 
        for i_p in range(par.Nh):
            for T_plus in par.T_boundles:
                sol.m[par.T-1,T_plus,:,i_p] = par.grid_m
                sol.c[par.T-1,T_plus,:,i_p] = par.grid_m
                sol.v[par.T-1,T_plus,:,i_p] = egm.util(sol.c[par.T-1,T_plus,:,i_p],T_plus,par)

        # Before last period
        # T_plus is time choice [T^w, T^H], e.g. [5, 10]
        for t in range(par.T-2,-1,-1):

            #Choice specific function
            for i_p, p in enumerate(par.grid_h):
                for i_t, T_plus in enumerate(par.T_boundles):

                    # Solve model with EGM
                    c,v = egm.EGM(sol,T_plus,p,t,par)
                    sol.c[t,i_t,:,i_p] = c
                    sol.v[t,i_t,:,i_p] = v
                
