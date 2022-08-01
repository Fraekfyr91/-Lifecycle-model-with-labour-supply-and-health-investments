import numpy as np
import tools

def EGM (sol,T_plus,p, t,par, health_cost = False, health_gains = False): 
    '''
    Solution to optimal consumption and values
    args:
        T_plus (list): Hours boundles of work and health improvements
        p (float): health state
        t (int): time
        health_cost (Bool): implement cost on bad health
    '''
    # find w_raw and averrage marginal utility in next period using working
    w_raw, avg_marg_u_plus = working(sol,T_plus,p,t,par, health_cost =False) 
    
    # Find marginal utility of bequests
    marg_u_bequest = marg_util_bequest(par.grid_a[t,:], par)
    # find propability of death
    delta = death_chance(t,p,par)
    
    # raw c, m and v
    c_raw = inv_marg_util((1-delta) * marg_u_bequest +par.beta*delta*par.R*avg_marg_u_plus,par)
    m_raw = c_raw + par.grid_a[t,:]
    
    # Upper Envelope
    c,v = upper_envelope(t,T_plus,c_raw,m_raw,w_raw,par, p)
    
    return c,v
    

def working(sol, T_plus, h, t, par,health_cost = False, health_gains = False):
    '''
    Find raw wage and average marginal utility in next period
    args:
        T_plus (list): Hours boundles of work and health improvements
        h (float): health state
        t (int): time
        health_cost (Bool): implement cost on bad health
    '''
    # Prepare
    xi = np.tile(par.xi,par.Na)
    
    
    h = np.tile(h, par.Na*par.Nxi)
    a = np.repeat(par.grid_a[t],par.Nxi) 
    w = np.tile(par.xi_w,(par.Na,1))

    # Next period states
    h_plus = (1 - par.phi1 + par.my * (T_plus[0]) / (par.Th_max) -par.phi2* t ** 2 ) * h  # health next period 
    h_plus = np.maximum(h_plus, par.eps) # cant have health below 0
    wage_plus = human_capital(t, h_plus,par) * xi # next period wage w, health effect on wage 
    m_plus = par.R * a + wage_plus * T_plus[1] / par.Tw_max # cash-on-hand next period
    m_plus += 3 * (t > 45) # flat subsidie if agent is above 70
    if health_cost:
        #cost =(h<par.sick)* np.minimum((np.exp(h_plus)/np.exp(10*h_plus))*par.ch_scale,par.ch_max) #monetary cost of health 
        cost =2
        m_plus -= cost # implement the cost
        if health_gains:
            gains = (h<par.sick)*h*par.gain_scale
            h_plus += gain # implement the gains
        
    m_plus = np.maximum(m_plus,0) # no borrowing 

    # Value, consumption, marg_util
    shape = (par.T_boundles.shape[0],m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)
    
    # range over possible hours next period
    for i, T_i in enumerate(par.T_boundles):
        # Choice specific value
        v_plus[i,:] = tools.interp_2d_vec(par.grid_m, par.grid_h, sol.v[t+1,i], m_plus, h_plus)
    
        # Choice specific consumption    
        c_plus[i,:] = tools.interp_2d_vec(par.grid_m, par.grid_h, sol.c[t+1,i], m_plus, h_plus)
        c_plus[i,:] = np.maximum(c_plus[i,:], par.eps)
        
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:], par)
        
    # Expected value
    V_plus, prob = logsum(v_plus, par.sigma_lambda) 
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nxi))
    w_raw = np.sum(w_raw,1) 
    marg_u_plus = np.sum(prob * marg_u_plus, axis = 0)

    #Expected  average marg. utility
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nxi))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)

    return w_raw, avg_marg_u_plus


def death_chance(t,h,par):
    '''
    chance of death
    args:
        t (int): time since start age
        h (float): health state
    return probability of death
    '''
    
    
    return 1 - 0.0006569 * (np.exp(0.1078507 * (t - 40)) - 1) *(1-h)


def human_capital(t, health,par):
    '''
    human capital function function
    args:
        t (int): time since start age
        h (float): health state
    returns: wage
    '''    
    age = t + par.start_age
    return np.exp(health + par.eta0 + par.eta1*age - par.eta2*age**2)


def upper_envelope(t,T_plus,c_raw,m_raw,w_raw,par, h):
    '''
    Upper envelope algorithm
    args:
        t (int): time since start age
        T_plus (list): Hours boundles of work and health improvements
        c_raw (numpy.ndarray): raw consumption array
        m_raw (numpy.ndarray): raw cash-on-hand array
        w_raw (numpy.ndarray): raw wage array
        h (float): health state
    return: consumption choices and values
        
    '''
    # Add a point at the bottom
    c_raw = np.append(1e-6,c_raw)  
    m_raw = np.append(1e-6,m_raw) 
    a_raw = np.append(0,par.grid_a[t,:]) 
    w_raw = np.append(w_raw[0],w_raw)

    # Initialize c and v   
    c = np.nan + np.zeros((par.Nm))
    v = -np.inf + np.zeros((par.Nm))
    
    # Loop through the endogenous grid
    size_m_raw = m_raw.size
    for i in range(size_m_raw-1):    
        
        c_now = c_raw[i] # set c_now
        m_low = m_raw[i] # set m_low
        m_high = m_raw[i+1] # set m_high
        
        c_slope = (c_raw[i+1]-c_now)/(m_high-m_low) # define slope
        
        w_now = w_raw[i] # set w_now
        a_low = a_raw[i] # set a low
        a_high = a_raw[i+1] # set a high
        w_slope = (w_raw[i+1]-w_now)/(a_high-a_low) # set w slope

        # Loop through the common grid
        for j, m_now in enumerate(par.grid_m):

            interp = (m_now >= m_low) and (m_now <= m_high) # interpolate m
            extrap_above = (i == size_m_raw-1) and (m_now > m_high) # extrappolate m

            if interp or extrap_above:
                # Consumption
                c_guess = c_now+c_slope*(m_now-m_low)
                c_guess = np.maximum(c_guess, 0.001)
                # post-decision values
                a_guess = m_now - c_guess
                w = w_now+w_slope*(a_guess-a_low)
                
                # Value of choice
                v_guess = util(c_guess,T_plus,t,par) + (1-death_chance(t,h,par)) *util_bequest(a_guess, par) + par.beta*death_chance(t,h,par)*w
                
                # Update
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess
    return c,v


def v(T, t, par):
    '''
    Disutility from working and helth improvements
    args:
        t (int): time since start age
        T_plus (list): Hours boundles of work and health improvements
    return: disutility value
    '''
    disutil = par.gamma_1[T[1]] + par.kappa_1*t**2*(T[0] / par.Th_max) *par.gamma_2[T[0]]*(T[0] / par.Th_max) + par.gamma_3 * (T[1] / par.Tw_max)
    return disutil

def util(c,T,t,par):
    '''
    Utility function
    args:
        c (float): consumption choice
        t (int): time since start age
        T_plus (list): Hours boundles of work and health improvements
    return utility
    '''

    # define leisure
    L = (par.Tw_max+par.Th_max - (T[1] + T[0])) / (par.Tw_max+par.Th_max)
    return (c**(1.0-par.rho)-1)/(1.0-par.rho) - v(T, t, par)+ par.l_scale * (L**(1-par.psi)/(1-par.psi))


def util_bequest(b, par): 
    '''
    Utility of bequests
    args:
        b (numpy.ndarray): a_guesses
    return Utility of bequests
    '''
    a = 0.001
    return par.b_scale * ((b + a)**(1-par.zeta_beq) - a**(1-par.zeta_beq)) / (1 - par.zeta_beq)
    
def marg_util_bequest(b, par):
    '''
    Marginal utility of bequests
    args:
        b (numpy.ndarray): a_guesses
    returns marginal utility of bequests
    '''
    
    a = 0.001
    return par.b_scale * (b + a)**(-par.zeta_beq) 
    

def marg_util(c,par):
    '''
    Marginal utilty
    args:
        c (float): consumption choice
    returns marginal utility
    '''
    return c**(-par.rho)


def inv_marg_util(u,par):
    '''
    inverted marginal utility
    args:
        u (float): marginal utility
    returns inverted marginal utility
    '''
    return u**(-1/par.rho)


def logsum(V, sigma):
    '''
    logaritmec sum function
    args:
        V (numpy.ndarray): Value array
        sigma (float): taste shock
    return expected value and propability
    '''
    
    # Maximum over the discrete choices (0 = for each column, i.e., for each "competing choice")
    mxm = V.max(0)

    # numerically robust log-sum
    log_sum = mxm + sigma*(np.log(np.sum(np.exp((V - mxm) / sigma),axis=0)))

    # d. numerically robust probability
    prob = np.exp((V- log_sum) / sigma)    
    
    return log_sum,prob