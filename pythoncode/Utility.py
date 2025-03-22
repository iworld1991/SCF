# ## Tools used for wealth distributions 
import numpy as np 

# + code_folding=[]
## lorenz curve
def lorenz_curve(grid_distribution,
                 pdfs,
                 nb_share_grid = 50):
    """
    parameters
    ======
    grid_distribution: grid on which distribution is defined
    pdfs: the fractions/pdfs of each grid ranges 
    
    return
    ======
    lc_vals: the fraction of people corresponding whose total wealth reaches the corresponding share, x axis in lorenz curve
    share_grids: different grid points of the share of total wealth, y axis in lorenz curve
    """
    total = np.dot(grid_distribution,pdfs)
    share_grids = np.linspace(0.0,0.99,nb_share_grid)
    share_cum = np.multiply(grid_distribution,pdfs).cumsum()/total
    lc_vals = []
    for i,share in enumerate(share_grids):
        where = min([x for x in range(len(share_cum)) if share_cum[x]>=share])
        this_lc_val = pdfs[0:where].sum()
        lc_vals.append(this_lc_val)
    return np.array(lc_vals),share_grids



# + code_folding=[]
## lorenz curve
def wealth_share(grid_distribution,
                 pdfs,
                 top_agents_share = 0.01):
    """
    parameters
    ======
    grid_distribution: grid on which distribution is defined
    pdfs: the fractions/pdfs of each grid ranges 
    top_agents_share: the top x share of agents for which wealth share is computed 
    
    return
    ======
    wealth_share: the fraction of wealth corresponding to the top x share of agents
    """
    total = np.dot(grid_distribution,pdfs)
    share_cum = np.multiply(grid_distribution,pdfs).cumsum()/total
    pdfs_cum = np.cumsum(pdfs) ## share of agents 
    where = min([x for x in range(len(pdfs_cum)) if pdfs_cum[x]>=(1-top_agents_share)])
    wealth_share = 1-share_cum[where]
    return wealth_share



## write a function that calculates SCF_lq_share_agents_ap, SCF_lq_share_ap from df, wgt, lqwealth

def get_lorenz_curve(df, var, wgt):
    """
    df: the dataframe
    wgt: the weight variable
    var: the variable of interest
    """
    var, weights = np.array(df[var]), np.array(df[wgt])
    var_sort_id = var.argsort()
    var_sort = var[var_sort_id]
    weights_sort = weights[var_sort_id]
    weights_sort_norm = weights_sort/weights_sort.sum()
    share_agents_ap, share_ap = lorenz_curve(var_sort,
                                             weights_sort_norm,
                                             nb_share_grid = 200)
    return share_agents_ap, share_ap

## also get_gini from df, wgt, var

def get_gini(df, var, wgt):
    """
    df: the dataframe
    wgt: the weight variable
    var: the variable of interest
    """
    share_agents_ap, share_ap = get_lorenz_curve(df, var, wgt)
    return gini(share_agents_ap, share_ap)

# -

# ## Gini coefficient 
#
# \begin{equation}
# \text{Gini} = 1- 2G\int^1_0 L(x)dx  
# \end{equation}
#
# where $L(x)$ is the lorenz function for x between $0$ to $1$.
#

# + code_folding=[0]
def gini(agents_share,
         value_share):
    """
    input
    =====
    agents_share: an array of fraction the agents from 0 to 1
    value_share: an array of value (wealth) share of the corresponding fraction of agents 
    
    output
    ======
    gini coefficients = B/(A+B) in lorenz curve where A+B = 1/2
    
    """
    agents_share_grid = agents_share[1:]-agents_share[:-1]
    gini = 1- 2*np.dot(value_share[1:],agents_share_grid)
    return gini 


# + code_folding=[0]
def h2m_ratio(a_grid,
              a_pdfs,
              cutoff):
    """
    input
    =====
    a_grid: an array of a grid: asset to permanent income ratio
    a_pdfs: an array of probabilities associated with these grids that sum up to one
    cutoff: the cutoff ratio for hands-to-month consumers, i.e. asset to income ratio below the cutooff is h2m
    
    output
    ======
    h2m_share: scalar indicating the share of h2m
    """
    h2m_where = np.where(a_grid<=cutoff)
    h2m_share = a_pdfs[h2m_where].sum()

    return h2m_share 

