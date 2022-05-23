import numpy as np
import tools

def EGM (sol,T_plus,p, t,par): 
    
    if T_plus[1] == 0:     #Retired =  Not working
        w_raw, avg_marg_u_plus = retired(sol,T_plus,p,t,par)
    else:               # Working
        w_raw, avg_marg_u_plus = working(sol,T_plus,p,t,par)

    #w_raw, avg_marg_u_plus = first_step(sol,T_plus,p,t,par)

    # raw c, m and v
    c_raw = inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    m_raw = c_raw + par.grid_a[t,:]
   
    # Upper Envelope
    c,v = upper_envelope(t,T_plus,c_raw,m_raw,w_raw,par)
    
    return c,v

def retired(sol, T_plus, h, t, par):
    # Prepare
    xi = np.tile(par.xi,par.Na)
    h = np.tile(h, par.Na*par.Nxi)
    a = np.repeat(par.grid_a[t],par.Nxi) 
    w = np.tile(par.xi_w,(par.Na,1))

    # Next period states
    h_plus = (1 - par.gamma + par.kappa * (T_plus[0]) / (1092) -0.02 * (t > 40) ) * h  # health next period 
    #wage_plus = human_capital(t, h_plus) * xi # next period wage w, health effect on wage 
    m_plus = par.R * a #+ wage_plus * T_plus[1] / 3000
    
    # Value, consumption, marg_util
    shape = (par.T_boundles.shape[0],m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)

    
    
    #hour_boundles = np.array([[0,0], [1000,0], [550,0]])
    # range over possible hours next period
    for i, T_i in enumerate(par.T_boundles):
        # Choice specific value
        v_plus[i,:] = tools.interp_2d_vec(par.grid_m, par.grid_h, sol.v[t+1,i], m_plus, h_plus)
    
        # Choice specific consumption    
        c_plus[i,:] = tools.interp_2d_vec(par.grid_m, par.grid_h, sol.c[t+1,i], m_plus, h_plus)
        c_plus[i,:] = np.maximum(c_plus[i,:], 0.001)
            
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:], par) 
    # value
    w_raw = tools.interp_2d_vec(par.grid_m, par.grid_h,sol.v[t+1,T_plus[1]], m_plus, h_plus)
    
    # Consumption
    c_plus = tools.interp_2d_vec(par.grid_m, par.grid_h,sol.c[t+1,T_plus[1]], m_plus, h_plus)
       
        
        
    # Expected value
    V_plus, prob = logsum(v_plus, par.sigma_eta) 
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nxi))
    w_raw = np.sum(w_raw,1) 
    marg_u_plus = np.sum(prob * marg_u_plus, axis = 0)

    #Expected  average marg. utility
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nxi))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)
    


    return w_raw, avg_marg_u_plus


def working(sol, T_plus, h, t, par):
    # Prepare
    xi = np.tile(par.xi,par.Na)
    h = np.tile(h, par.Na*par.Nxi)
    a = np.repeat(par.grid_a[t],par.Nxi) 
    w = np.tile(par.xi_w,(par.Na,1))

    # Next period states, T_i[0] is exercise, T_i[1] is work
    
    h_plus = (1 - par.gamma + par.kappa * (T_plus[0]) / (1092) -0.01 * (t > 40) ) * h  # health next period 
    wage_plus = human_capital(t, h_plus) * xi # next period wage w, health effect on wage 
    m_plus = par.R * a + wage_plus * T_plus[1] / 3000

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
        c_plus[i,:] = np.maximum(c_plus[i,:], 0.001)
            
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:], par) 
       
    # Expected value
    V_plus, prob = logsum(v_plus, par.sigma_eta) 
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nxi))
    w_raw = np.sum(w_raw,1) 
    marg_u_plus = np.sum(prob * marg_u_plus, axis = 0)

    #Expected  average marg. utility
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nxi))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)

    return w_raw, avg_marg_u_plus
def human_capital(t, health):
    # see plot, increases with age up untill a certain point
    
    inc0=0.75
    inc1=0.04
    inc2=0.0003
    
    age = t + 19
    return( (2*health) * np.exp(inc0 + inc1*age - inc2*age**2))




def upper_envelope(t,T_plus,c_raw,m_raw,w_raw,par):
    
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

        c_now = c_raw[i]        
        m_low = m_raw[i]
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_now)/(m_high-m_low)
        
        w_now = w_raw[i]
        a_low = a_raw[i]
        a_high = a_raw[i+1]
        w_slope = (w_raw[i+1]-w_now)/(a_high-a_low)

        # Loop through the common grid
        for j, m_now in enumerate(par.grid_m):

            interp = (m_now >= m_low) and (m_now <= m_high) 
            extrap_above = (i == size_m_raw-1) and (m_now > m_high)

            if interp or extrap_above:
                # Consumption
                c_guess = c_now+c_slope*(m_now-m_low)
                c_guess = np.maximum(c_guess, 0.001)
                # post-decision values
                a_guess = m_now - c_guess
                w = w_now+w_slope*(a_guess-a_low)
                
                # Value of choice
                v_guess = util(c_guess,T_plus, t,par)+par.beta*w
                
                # Update
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess
    return c,v

# disutility from working fun
def v(T, t, par):
    work_h = T[1]
    exercise_h = T[0]
    gamma_1 = {0: 0, 1000: 1.4139, 2000: 2.0088, 2250: 2.9213, 2500: 2.8639, 3000: 3.8775}
    gamma_2 = 0.2
    gamma_3 = 0.3
    
    # disutility from working and exercise that increaes with age -> hopefully it will make agents stop working when old, and make exercise more costly
    kappa_2 = 0.00004 
    kappa_1 = 0.00008
    
    return gamma_1[work_h]*0.2 + kappa_1 * (t-40)**2 * (t > 40) + gamma_2 * (exercise_h > 0) + gamma_3 * (exercise_h > 500) + kappa_2 * (t-40) ** 2 * (t > 40) * (exercise_h > 0)

def util(c,T,t,par):
    work_h = T[1]
    exercise_h = T[0]

    return (c**(1.0-par.rho))/(1.0-par.rho) - 2*v(T, t, par)


def marg_util(c,par):
    return c**(-par.rho)


def inv_marg_util(u,par):
    return u**(-1/par.rho)


def logsum(V, sigma):
    # Maximum over the discrete choices (0 = for each column, i.e., for each "competing choice")
    mxm = V.max(0)

    # numerically robust log-sum
    log_sum = mxm + sigma*(np.log(np.sum(np.exp((V - mxm) / sigma),axis=0)))

    # d. numerically robust probability
    prob = np.exp((V- log_sum) / sigma)    

    return log_sum,prob