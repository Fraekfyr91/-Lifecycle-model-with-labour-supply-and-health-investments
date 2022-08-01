# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim as egm
from numba import njit, int64, double, boolean, int32,void
from scipy import optimize
import copy

class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par

        par.T = 70 # Number of years 

        # Model parameters
        
        par.rho = 0.8 # CRRA coefficient in utillity
        
        par.beta = 0.96 # Discount factor
        
        #par.alpha = 0.75
        
        par.R = 1.04 # returns from assets
        
        #par.W = 1
        par.sigma_wage = 0.05 # variance of wage shocks
        par.sigma_lambda = 0.1 # variance taste shocks
               
        par.l_scale = 0.5 #leisure scale
        
        par.disutil_scale = 0.5 # disutility scale
        
        par.penst = 45 #pension age = number of years as worker
        
        par.pens = 3 # pension subsidie
     
        par.eps = 0.00000001 #small number > 0
        
        par.b_scale =0.68659 # bequest motive scale parameter ishkarows parameter 
        
        par.zeta_beq = 0.48834 # CRRA coefficient in utility of bequest
        
        par.eta0=0.75 # base income term
        par.eta1=0.04 # age dependent term
        par.eta2=0.0003 # age squered income term
        
        
        par.start_age =25#Start age of agents
        
        par.ch_scale = 5000 # scale to health cost function
        
        par.ch_max = 20 # max health expenditure
        
        par.sick = 0.75 # health cost threshold
        
        par.gain_scale =0.02 # gain to health if below threshold
        
        par.my = 0.05   # abilty to transfer exercise into health
        
        par.phi1 = 0.05  # health decay
        
        par.phi2 = 0.00005 # age squared health decay dependent term

        # parameters for disutillity 
        #par.kappa_1 = 0.0005 # working cost relative to max working load
        par.kappa_1 = 0.00035
        #par.kappa_2 = 0.00035 # squared term cost of exercise
        
        par.gamma_1 = {0: 0, 1000: 0.5, 2000: 1, 2250: 1.1, 2500: 1.15, 3000: 1.35}
        par.gamma_2 = {0: 0, 500: 0.425, 1000: 0.6}
        par.gamma_3 = 0.15
        
        par.psi = 0.5 # leisure 
        
        par.Th_max = 1000 # max hours of health
        
        par.Tw_max = 3000 # max hours of work
        
        

        # Grids and numerical integration
        par.m_max = 50
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 50
        par.a_phi = 1.1  # Curvature parameters
        par.h_max = 1.0
        par.p_phi = 1.0 # Curvature parameters

        par.Nxi = 8
        par.Nm = 150
        par.Na = 150
        par.Nh = 100

        par.NTh = 7 # number of possible exercise bundles 
        par.Nm_b = 50
        par.Rh = 1
        # simulation
        par.sim_mini = 2.5 # initial m in simulation
        par.simN = 100 # number of persons in simulation
        par.simT = par.T# number of periods in simulation
        
     
    def create_hour_boundles(self):
        '''
        Crete hours boundles
        args: None
        Return Array of list with a bunch of time spend on work of health improvements
        '''
        return np.array([[0,0], [500, 0], [1000, 0], [500, 1000], [500, 2000], [1000, 1000],[1000, 2000],[500,2500],[1000,2500], [1000, 3000]])
    def create_grids(self):
        '''
        Create grids for:
        Cash-on-hand
        Next periods assets
        Health states
        Time boundles
        
        '''
        # Set seed
        np.random.seed(2020)
        par = self.par # define parameters
        # Check parameters
        assert (par.rho >= 0), 'not rho > 0'
        # Shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_wage,par.Nxi)
        # scale shocks
        #par.psi = par.xi * 0.02 
        
        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)]) 

        # Health states
        par.grid_h = tools.nonlinspace(0+1e-4,par.h_max,par.Nh,par.p_phi)
        par.T_boundles = self.create_hour_boundles()
        par.NT = par.T_boundles.shape[0]

        par.Nshocks = par.xi * par.Na
        
    
                 
    def solve(self, print_iteration_number = True, health_cost =False, health_gains = False):
        '''
        Model Solver
        args:
            print_iteration (Bool): Prints current period
            health_cost (Bool): Agents pays money for declining health
        '''
        
        # Initialize
        par = self.par
        sol = self.sol
        shape=(par.T, par.NT, par.Nm, par.Nh)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        def obj(c,m,T,t, par):
            b= m-c
            return -(egm.util(c,T,t,par)+egm.util_bequest(b,par))
        # Last period
        for i_h in range(par.Nh):
            #sol.m[par.T-1,i_h,:,i_h] = par.grid_m # grid for cash
            for i_c, m in enumerate(par.grid_m):
                
                c= optimize.minimize_scalar(obj, args=(m,[0,0],par.T-1,par),bounds =(0,m), method ='bounded').x
                

                sol.c[par.T-1,:,:,i_h] = c

                sol.v[par.T-1,:,:,i_h] = egm.util(sol.c[par.T-1,:,:,i_h],[0,0],par.T-1,par) +egm.util_bequest(m-c,par) # grid for value
        
        # Before last period
        for t in range(par.T-2,-1,-1):
            if print_iteration_number:
                print(f'Evaluating period: {t}')
            #Choice specific function
            for i_h, p in enumerate(par.grid_h):
                for i_t, T_plus in enumerate(par.T_boundles):
                    # Solve model with EGM
                    c,v = egm.EGM(sol,T_plus,p,t,par,health_cost, health_gains)
                    sol.c[t,i_t,:,i_h] = c # solve for consumption
                    sol.v[t,i_t,:,i_h] = v # solve for value
                    
                    
   
    def simulate (self, health_cost=False, health_gains = False):
        seed = np.random.seed(999)
     

        sol = self.sol
        par = self.par
        par.simN = 1000
        
        sim= self.sim

        # Initialize
        shape = (par.simT, par.simN)
        sim.m = np.nan +np.zeros(shape)
        sim.c = np.nan +np.zeros(shape)
        sim.a = np.nan +np.zeros(shape)
        sim.h = np.nan +np.zeros(shape)
        sim.v = np.nan +np.zeros(shape)

        sim.Th = np.nan +np.zeros(shape)
        sim.Tw = np.nan +np.zeros(shape)
        sim.income = np.nan +np.zeros(shape)
        sim.wage = np.nan +np.zeros(shape)

        sim.y = np.nan +np.zeros(shape)

        par.Nshocks = par.xi.size * par.Na
        # Shocks

        # 0.05
        par.xi,par.xi_w = tools.GaussHermite_lognorm(0.05,par.Nxi)

        shock = np.random.lognormal(mean = -0.5*par.sigma_wage, sigma = par.sigma_wage, size = (par.T,par.simN))
        shock2 = np.random.lognormal(mean = -0.5*par.sigma_wage, sigma = par.sigma_wage, size = (par.T,par.simN)) * 0.02

        #shock2 = 1
        #shock = np.random.choice(par.xi, (par.T,par.simN),replace=True,p=par.xi_w) #draw values between 0 and Nshocks-1, with probability w

        sim.xi = shock

        # Initial values
        sim.m[0,:] = np.random.normal(10, 1, par.simN) 
        sim.h[0,:] = np.random.uniform(0.9, 1, par.simN) 
        #sim.h[0,:] =  1
        # Simulation 
        for t in range(par.simT):
            V = np.zeros( (self.par.NT, self.par.simN) )
            C = np.zeros( (self.par.NT, self.par.simN) )
            for i, T_i in enumerate(self.par.T_boundles): # possible hour choices

                #print( np.average(  sol.v[t, i ] ))
                C[i, :] = tools.interp_2d_vec(par.grid_m, par.grid_h, sol.c[t, i], sim.m[t, :], sim.h[t, :])

                V[i, :] = tools.interp_2d_vec(par.grid_m , par.grid_h, sol.v[t, i ], sim.m[t, :], sim.h[t, :]) # value for every choice


            #print( np.average( sol.v[t, 0] ))

            h_i = V.argmax(0) # index of best choice

            VV, VV_w  = egm.logsum(V, self.par.sigma_lambda)
            base = np.arange(0,self.par.T_boundles.shape[0])

            # dont know which way is fastest -> loop or "apply_along_axis
            # this way seems more efficient
            h_i = np.zeros((par.simN), dtype = int)
            for i in range(VV_w.shape[-1]):
                h_i[i] = np.random.choice(base,  p = VV_w[:,i] )

            def test(x):
                return np.random.choice(base,  p = x )
            #h_i = np.apply_along_axis(test, 0, VV_w)

            #print(h_i.shape)

            sim.Th[t, :] = self.par.T_boundles[h_i][:, 0]
            sim.Tw[t, :] = self.par.T_boundles[h_i][:, 1]
            if t == par.T-1:
                sim.Th[t, :] = 0
                sim.Tw[t, :] = 0

            # egm.logsum
            temp = np.arange(self.par.simN)

            Cs = C[h_i, temp] # magic to get all the best choices

            sim.c[t, :] = Cs  
            sim.c[t, :] = np.minimum(sim.c[t, :], sim.m[t, :])
            sim.c[t, :] = np.maximum(sim.c[t, :], 1)
            #print(np.array([sim.Th[t, :]] + sim.Tw[t, :]))
            #sim.v[t, :] = egm.util(sim.c[t, :], [sim.Th[t, :], sim.Tw[t, :]], t, par)
            sim.a[t,:] = sim.m[t,:] - sim.c[t,:]
            if t< par.simT-1:

                sim.h[t+1,:] = (1 - self.par.phi1 + self.par.phi1 * (sim.Th[t,:] / (1000) -0.00005 * t ** 2 )) * sim.h[t, :]

                sim.h[t+1,:] = np.maximum(sim.h[t+1,:], 0) # cant have health below 0

                wage = egm.human_capital(t, sim.h[t, :], par) * sim.xi[t, :] # next period wage w. health effect on wage 
                sim.wage[t, :] = wage
                sim.income[t,:] = wage * (sim.Tw[t, :]) / (3000)
                m_plus = par.R * sim.a[t, :] + sim.income[t,:]
                m_plus += 3 * (t > 45)


                if health_cost:
                    cost = (sim.h[t+1,:]<0.8)*np.minimum(np.exp(sim.h[t+1,:])/np.exp(10*np.exp(sim.h[t+1,:]))*500,20) #monetary cost of health 


                    m_plus -= 2*(sim.h[t+1,:]<par.sick)

                    if health_gains:
                        gain = (sim.h[t+1,:]<par.sick)* (sim.h[t+1,:] * par.gain_scale)
                        sim.h[t+1,:] += gain
                    #m_plus = m_plus - np.minimum((np.exp(sim.h[t+1,:])/np.exp(10*sim.h[t+1,:]))*500, 20)


                sim.h[t+1,:] = np.minimum(sim.h[t+1,:], 1)
                sim.m[t+1,:] = m_plus
                sim.m[t+1,:] = np.maximum(m_plus, 0)


        return sim
