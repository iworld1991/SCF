# ## Tools used for wealth distributions 
import numpy as np 
import matplotlib.pyplot as plt

# + code_folding=[]
## lorenz curve
def lorenz_curve(grid_distribution,
                 pdfs,
                 nb_share_grid = 50):
    """
    Computes the inverse Lorenz curve, i.e., the cumulative share of the population 
    required to account for a given share of total wealth.

    Parameters
    ----------
    grid_distribution : array-like
        Grid over which the distribution is defined (e.g., wealth levels).
        Should be sorted in ascending order (or will be treated as such).
    
    pdfs : array-like
        Probability mass function or density weights corresponding to each point
        in the grid. Must sum to 1 (or will be treated as such).

    nb_share_grid : int, optional (default=50)
        Number of points to compute along the wealth share axis (y-axis of Lorenz curve).

    Returns
    -------
    lc_vals : np.ndarray
        The cumulative share of the population whose total wealth adds up 
        to the given share of total wealth (x-axis → y-axis mapping of inverse Lorenz).

    share_grids : np.ndarray
        Array of total wealth share values (from 0 to ~0.99), representing the 
        x-axis of the Lorenz curve.

    Notes
    -----
    This function effectively computes the **inverse** of the standard Lorenz curve.
    That is, for each target wealth share, it returns the population share needed to 
    accumulate up to that point in the distribution. It's useful when one is interested 
    in questions like: "What fraction of people owns the bottom 30% of wealth?"
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


def inverse_lorenz_curve(grid_distribution, pdfs, nb_pop_grid=50):
    """
    parameters
    ==========
    grid_distribution : array-like
        Grid values (e.g., wealth levels)
    pdfs : array-like
        Probability mass function or distribution weights
    nb_pop_grid : int
        Number of equally spaced population share points (x-axis of Lorenz)

    returns
    =======
    pop_shares : np.ndarray
        Population shares (x-axis)
    wealth_shares : np.ndarray
        Corresponding cumulative wealth shares (y-axis)
    """
    # Sort the distribution by grid (e.g., ascending wealth)
    sort_idx = np.argsort(grid_distribution)
    sorted_grid = np.array(grid_distribution)[sort_idx]
    sorted_pdfs = np.array(pdfs)[sort_idx]

    # Total wealth
    total_wealth = np.dot(sorted_grid, sorted_pdfs)

    # Cumulative population share
    pop_cum = sorted_pdfs.cumsum()
    
    # Cumulative wealth share
    wealth_cum = np.multiply(sorted_grid, sorted_pdfs).cumsum() / total_wealth

    # Generate x-axis (population shares)
    pop_shares = np.linspace(0.0, 0.99, nb_pop_grid)
    wealth_shares = []

    for p in pop_shares:
        # Find first index where cumulative population >= p
        where = min([i for i in range(len(pop_cum)) if pop_cum[i] >= p])
        w = wealth_cum[where]
        wealth_shares.append(w)

    return pop_shares, np.array(wealth_shares)


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

def get_lorenz_curve(df, var, wgt,nb_share_grid = 200,how='wealth_to_population'):
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
    if how =='wealth_to_population':
        share_pop, share_wealth = lorenz_curve(var_sort,
                                                weights_sort_norm,
                                                nb_share_grid = nb_share_grid)
    elif how=='population_to_wealth':
        share_pop, share_wealth = inverse_lorenz_curve(var_sort,
                                                        weights_sort_norm,
                                                        nb_pop_grid = nb_share_grid)
    return share_pop, share_wealth

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

## two other functions that give Lorenz curve more details 

def weighted_percentiles(data, variable, weights, percentiles = [], 
                         dollar_amt = False, subgroup = None, limits = []):
    """
    data               specifies what dataframe we're working with
    
    variable           specifies the variable name (e.g. income, networth, etc.) in the dataframe
    
    percentiles = []   indicates what percentile(s) to return (e.g. 90th percentile = .90)
    
    weights            corresponds to the weighting variable in the dataframe
    
    dollar_amt = False returns the percentage of total income earned by that percentile 
                       group (i.e. bottom 80% of earners earned XX% of total)
                         
    dollar_amt = True  returns the $ amount earned by that percentile (i.e. 90th percentile
                       earned $X)
                         
    subgroup = ''      isolates the analysis to a particular subgroup in the dataset. For example
                       subgroup = 'age' would return the income distribution of the age group 
                       determined by the limits argument
                       
    limits = []        Corresponds to the subgroup argument. For example, if you were interesting in 
                       looking at the distribution of income across heads of household aged 18-24,
                       then you would input "subgroup = 'age', limits = [18,24]"
                         
    """
    a  = list()
    data[variable+weights] = data[variable]*data[weights]
    if subgroup is None:
        tt = data
    else:
        tt = data[data[subgroup].astype(int).isin(range(limits[0],limits[1]+1))] 
    values, sample_weight = tt[variable], tt[weights]
    
    for index in percentiles: 
        values = np.array(values)
        index = np.array(index)
        sample_weight = np.array(sample_weight)

        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_percentiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_percentiles /= np.sum(sample_weight)
        a.append(np.interp(index, weighted_percentiles, values))
    
    if dollar_amt is False:    
        return[tt.loc[tt[variable]<=a[x],
                      variable+weights].sum()/tt[variable+weights].sum() for x in range(len(percentiles))]
    else:
        return a
    
    
def figureprefs(data, 
                variable = 'income', 
                labels = False, 
                legendlabels = []):
    
    percentiles = [i * 0.05 for i in range(20)]+[0.99, 1.00]

    fig, ax = plt.subplots(figsize=(6,6));

    ax.set_xticks([i*0.1 for i in range(11)]);       #Sets the tick marks
    ax.set_yticks([i*0.1 for i in range(11)]);

    vals = ax.get_yticks()                           #Labels the tick marks
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);
    ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals]);

    ax.set_title('Lorenz Curve: United States in 2016',  #Axes titles
                  fontsize=18, loc='center');
    ax.set_ylabel('Cumulative Percent', fontsize = 12);
    ax.set_xlabel('Percent of Agents', fontsize = 12);
    
    if type(data) == list:
        values = [weighted_percentiles(data[x], variable,
                    'wgt', dollar_amt = False, percentiles = percentiles) for x in range(len(data))]
        for index in range(len(data)):
            plt.plot(percentiles,values[index],
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels[index]);
            for num in [10, 19, 20]:
                ax.annotate('{:3.1f}%'.format(values[index][num]*100), 
                    xy=(percentiles[num], values[index][num]),
                    ha = 'right', va = 'center', fontsize = 12);

    else:
        values = weighted_percentiles(data, variable,
                    'wgt', dollar_amt = False, percentiles = percentiles)
        plt.plot(percentiles,values,
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels);

    plt.plot(percentiles,percentiles, linestyle =  '--', color='k',
            label='Perfect Equality');
   
    plt.legend(loc = 2)