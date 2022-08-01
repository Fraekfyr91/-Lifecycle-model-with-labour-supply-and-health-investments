import numpy as np
from numba import njit, int64, double, boolean, int32,void
import math

# interpolation functions:
@njit(int64(int64,int64,double[:],double))
def binary_search(imin,Nx,x,xi):
        
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

@njit(double(double[:],double[:],double))
def interp_linear_1d_scalar(grid,value,xi):
    """ raw 1D interpolation """

    # a. search
    ix = binary_search(0,grid.size,grid,xi)
    
    # b. relative positive
    rel_x = (xi - grid[ix])/(grid[ix+1]-grid[ix])
    
    # c. interpolate
    return value[ix] + rel_x * (value[ix+1]-value[ix])

@njit
def interp_linear_1d(grid,value,xi):

    yi = np.empty(xi.size)

    for ixi in range(xi.size):

        # c. interpolate
        yi[ixi] = interp_linear_1d_scalar(grid,value,xi[ixi])
    
    return yi

@njit(double(double[:],double[:],double[:,:],double,double,int32,int32),fastmath=True)
def _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2):
    """ Code is from: https://github.com/NumEconCopenhagen/ConsumptionSaving, and the package can be downloaded
    
    2d interpolation for one point with known location
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
    Returns:
        yi (double): output
    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right                    
            nom += nom_1*nom_2*value[j1+k1,j2+k2]

    return nom/denom


@njit(double(double[:],double[:],double[:,:],double,double),fastmath=True)
def interp_2d(grid1,grid2,value,xi1,xi2):
    """ Code is from: https://github.com/NumEconCopenhagen/ConsumptionSaving, and the package can be downloaded


    2d interpolation for one point
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point
    Returns:
        yi (double): output
    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)

    return _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2)


@njit(double[:](double[:],double[:],double[:,:],double[:],double[:]),fastmath=True)
def interp_2d_vec(grid1,grid2,value,xi1,xi2):
    """ Code is from: https://github.com/NumEconCopenhagen/ConsumptionSaving, and the package can be downloaded

    
    2d interpolation for vector of points
        
    Args:
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (numpy.ndarray): input vector
        xi2 (numpy.ndarray): input vector
        yi (numpy.ndarray): output vector
    """
    shape = (xi1.size)
    yi = np.nan+np.zeros(shape)

    for i in range(xi1.size):
        yi[i] = interp_2d(grid1,grid2,value,xi1[i],xi2[i])

    return yi


# State space
def nonlinspace(x_min, x_max, n, phi):
    """ like np.linspace between with unequal spacing
    phi = 1 -> eqaul spacing
    phi up -> more points closer to minimum
    """
    assert x_max > x_min
    assert n >= 2
    assert phi >= 1
 
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi
    
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y

def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w

def GaussHermite_lognorm(sigma,n):

    x, w = gauss_hermite(n)
    x = np.exp(x*math.sqrt(2)*sigma - 0.5*sigma**2)
    w = w / math.sqrt(math.pi)

    # assert a mean of one
    assert(1 - np.sum(w*x) < 1e-8 ), 'The mean in GH-lognorm is not 1'
    return x, w

def modelcurves(Vars, range_, label, model, b=False):
    '''
    Find averages af model values
    args:
        models (list): list of model simulations
    return Model values
    '''
    
    # containers
    base =[]
    alt1 = []
    alt2 = []
    
    #loop over simulation results
    for i in range(len(Vars)):
        # loop over years
        for j in range(range_):
            val = Vars[i][j, :] 
            val = np.average(val)
            if i == 0:
                base.append(val) # store values from first model
            elif i == 1:
                alt1.append(val) # store values from secound model
            elif i == 2:
                alt2.append(val) # store values from third model
            else:
                print('no list available') # øv
                
                
    fig = plt.figure(figsize=(8,6))
    fig.tight_layout()
    ax = fig.add_subplot(111)
    x= range(25,25+range_)
    if len(Vars) == 1:
        ax.plot(x,base, color ='blue') # plot base
    else:
        # create arrays
        basearray = np.array(base)
        alt1array = np.array(alt1)
        alt2array = np.array(alt2)
        # get difference
        delta1 = np.subtract(base, alt1array)
        delta2 = np.subtract(base, alt2array)
        # difference to lists
        subtracted1 = list(delta1)
        subtracted2 = list(delta2)
        
        #plot simulations and differences
        ax.plot(x,base,color ='blue')
        ax.plot(x,alt1, ':', color = 'green')
        ax.plot(x,alt2,'--', color = 'coral')
        ax.bar(x,height = subtracted1, color = 'green')
        ax.bar(x,height = subtracted2,color = 'coral')
    
    # create title
    ax.set_ylim(top=max(base)+max(base)*0.05 )
    ax.set_xticks([x for x in range(20,91,10)], minor=True)
    ax.set_yticks(np.linspace(0,0.8,5), minor=True)
    
    ax.set_clip_on(False)
    
    ax.set_ylabel(label)
    ax.set_xlabel('age')
    ax.margins(x=0.02,)
    ax.margins(y=0.0)
    ax.set_xticks([x for x in range(25,95,10)], minor=True)
    ax.set_yticks(np.linspace(0,1,10), minor=True)
    ax.set_clip_on(False)
    #plt.title(label, fontsize=15, fontweight='bold')
    legend = plt.legend(model, frameon = 1, prop={'size': 14}, loc='lower left')
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('grey')

    
    if b ==True:
        label =f'{label}_baseline'
    elif b == 'sim':
        label =f'{label}_sim'
        
    elif b == 'sim2':
        label =f'{label}_sim2'
        ax.set_ylim(top=max(base)+max(base)*0.1 )
    fig_name = label.replace(' ','_') 
    plt.savefig(f'{fig_name}_figure.png')
    plt.show()

 