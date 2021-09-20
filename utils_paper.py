import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm
from scipy.sparse import lil_matrix, linalg
from sklearn.cluster import AffinityPropagation, Birch, SpectralClustering, MeanShift
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, KMeans
import seaborn as sns
import fssa
from scipy.stats import chisquare
    
## Building the Hamiltonian 
def binaryConvert(x=5, L=4):
	'''
	Convert base-10 integer to binary and adds zeros to match length
	_______________
	Paramters:
		x : base-10 integer
		L : length of bitstring 
	_______________
	returns: 
		b : Bitstring 
	'''
	b = bin(x).split('b')[1]
	while len(b) < L:
		b = '0'+b
	return b

def nOnes(bitstring='110101'):
	'''
	Takes binary bitstring and counts number of ones
	_______________
	Parameters:
	bitstring : string of ones and zeros
	_______________
	returns: 
		counta : number of ones
	'''
	counta = 0
	for i in bitstring:
		if i=='1':
			counta += 1
	return counta

def binomial(n=6, pick='half'):
	'''
	find binomial coefficient of n pick k,
	_______________
	Parameters:
		n : total set
		pick : subset 
	_______________
	returns:
		interger
	'''
	if pick == 'half':
		pick = n//2

	return int(factorial(n) / factorial(n-pick) / factorial(pick))

def basisStates(L=5):
	'''
	Look for basis states
	_______________
	Parameters:
		L : size of system; integer divible by 2
		
	_______________
	Returns:
		dictionaries (State_to_index, index_to_State)
	'''
	if L%2!=0:
		print('Please input even int for L')

	s2i = {} # State_to_index
	i2s = {} # index_to_State

	index = 0
	for i in range(int(2**L)): # We could insert a minimum
		binary = binaryConvert(i, L)
		ones = nOnes(binary)

		if ones == L//2:
			s2i[binary] = index
			i2s[i] = binary
			index +=1

	return (s2i, i2s)

def energyDiagonal(bitString='010', V=[0,0.5,1] , U=4.3):
	'''
	Diagonal of Hamiltonian with periodic boundary conditions 
	______________
	Parameters:
		bitString : ones and zeros; string 
		V : onsite potentials for each site; list of floats
		U : interaction; float
	______________
	returns :
		E : diagonal of H; list of floats
	'''
	E = 0
	for index, i in enumerate(bitString):
		if i =='1':
			E += V[index]
			try:
				if bitString[index+1] == '1':
					E += U

			except IndexError:
				if bitString[0] == '1':
					E += U
	return E

def constructHamiltonian(L = 4, W = 2, U = 1, t = .42, method='dense', seed=42):
	'''
	Constructs the Hamiltonian matrix
	________________
	Parameters:
		L : size of system; integer divible by 2
		W : disorder strength; float
		U : Interaction; flaot
		t : hopping term; float
		method : 'sparse' or 'dense'
		seed : seed for random
	________________
	returns:
		Sparse Hamiltonian
	'''
	np.random.seed(seed)
	V = np.random.uniform(-1,1,size=L) * W
	num_states = binomial(L)

	(s2i, i2s) = basisStates(L)
	if method.lower() == 'dense':
		H = np.zeros((num_states,num_states))
	elif method.lower() == 'sparse':
		H = lil_matrix((num_states,num_states))
	else:
		print("no valid method; input 'dense' or 'sparse'")
		return 0

	for key in s2i.keys():
		H[s2i[key],s2i[key]] = energyDiagonal(key, V, U)  # fill in the diagonal with hop hopping terms
		for site in range(L):
			try:
				if (key[site] == '1' and key[site+1]== '0'):
					new_state = key[:site] + '0' + '1' + key[site+2:]
					H[s2i[new_state], s2i[key]], H[s2i[key], s2i[new_state]] = t ,t

			except IndexError: # periodic boundary conditions
				if (key[site] == '1' and key[0]== '0'):
					new_state = '1' + key[1:site] + '0'
					H[s2i[new_state], s2i[key]], H[s2i[key], s2i[new_state]] = t ,t
	return H

def diag(Hamiltonian):
	'''
	Diagonalizes Hamiltonian, dense or sparse
	_______________
	Parameters:
		Hamiltonian
	_______________
	Returns:
		eigenvalues
	'''
	try:
		return np.linalg.eigh(Hamiltonian)
	except np.linalg.LinAlgError:
		k = np.shape(Hamiltonian)[0]
		eigvals, eigvecs =  linalg.eigsh(Hamiltonian, k= k-1)
		return eigvals, eigvecs

def buildDiagSave(L = 10, seeds = 10, 
	min_disorder = 0.5, max_disorder = 5.5, steps = 11,
    method = 'dense', location = 'data'):
    '''
    builds Hamiltonians, diagonalizes and saves eigvecs and eigvals    
    '''
    for w in tqdm(np.linspace(min_disorder,max_disorder,steps)):
        for seed in range(seeds):
            H = constructHamiltonian(L = L, W = w, seed=seed, method=method)
            eigvals, eigvecs = diag(H)
            np.savez(location+'/results-L-{}-W-{}-seed-{}.npz'.format(L, round(w,2), seed, method) ,eigvals, eigvecs)

## Loading data
def list_filenames(
    location = 'data/',
    Ls = [8],
    ws = [1.0,2.5,4.0,5.5,7.0],
    num_seeds = 10):
    seeds = np.arange(num_seeds)
    
    Filenames = [[[location+'results-L-{}-W-{}-seed-{}.npz'.format(L,w,2) for seed in seeds] for w in ws] for L in Ls]
    return Filenames

def load_eigs(filename='data/results-L-2-W-0.1-seed-0.npz',
	vecs_or_vals = 'vals'):
	'''Load eigen-vals/vecs correponsing to a given filename.'''
	data = np.load(filename)
	if vecs_or_vals == 'vals':
		out = data[data.files[0]]
	elif vecs_or_vals == 'vecs':
		out = data[data.files[1]]
	return out

def load_many_eigs(filenames, vecs_or_vals = 'vals'):
    '''Load many eigen-vals/vecs.'''
    eigs = [[[load_eigs(file, vecs_or_vals) for file in file_w]for file_w in file_L]for file_L in filenames]
    return np.array(eigs)

def count_lower_than(lst, lim):
    counta = 0
    for i in lst:
        if i < lim:
            counta +=1
    return counta


def nn2(data, plot=False, return_xy = False,eps=.01, xshift=False, del_vals=2):
    '''
    Find intrinsic dimension (ID) via 2-nearest-neighbours

    https://www.nature.com/articles/s41598-017-11873-y
    https://arxiv.org/pdf/2006.12953.pdf
    _______________
    Parameters:
        eigvecs
        plot : create a plot; boolean; dafault=False
    _______________
    Returns:
        m : Slope
    
    '''

    N = len(data)
    
    distance_matrix = np.zeros((N, N))
    # Making the distance matrix: distance from each eigvec to all others
    for i, eigvec1 in enumerate(data):
        for j, eigvec2 in enumerate(data):
            if j <= i:
                pass
            else:
                distance = sum(abs((eigvec1-eigvec2)))
                distance_matrix[i,j], distance_matrix[j,i] = distance, distance
        #print(distance_matrix) # To see how it fills √

    # table of distances - state and \mu= r_2/r_1
    mu = np.zeros((N,2))
    for index, line in enumerate(distance_matrix):
        r1, r2 = sorted(line)[1:3]
        mu[index,0] = index+1
        mu[index,1] = r2/(r1+eps)
    if xshift == True:
        mu[:,1] -= min(mu[:,1])+1
        
    #permutation function
    sigma_i = dict(zip(range(1,len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))
    mu = dict(mu)
    #cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i/N

    #fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])

    #avoid having log(0)
    x = x[y>0]
    y = y[y>0]
    

	

    y = -1*np.log(y)
    #y2 = -1*np.log(y2)
	
    #fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]

    if plot==True:
        #fig, ax = plt.subplots(2,1, sharex=True)
        plt.scatter(x,y, c='g')
        plt.plot(x,x*d, c='r', ls='--')
        
      #ax[plot_index].set_xlabel('log($\mu$)', fontsize=12)
        #plt.text(0,5,'w={}'.format(w), fontsize=13)
        #plt.text(0,4.5,'d={}'.format(round(d,3)), fontsize=13)
        
            
    if return_xy:
        return x,y, d#,x2,y2
    else:
        return d

def L2_loss_linear_originBound(X,Y, slope):
    L = np.zeros(len(X))
    index=0
    for x,y in zip(X,Y):
        L[index] = (y-x*slope)**2
        index+=1
    return sum(L)

def weigt_from_loss(loss):
    return 1/loss

def scale_collapse2(data, ws, l = [8,10,12],rho_c0=3.5,
	nu0=2., zeta0=2., skip_initial = 2, drop_ls = 0):
    
    a = data
    da = a / 100

    res = fssa.autoscale(l=l, rho=ws, a=a, da=da, rho_c0=rho_c0, nu0=nu0, zeta0=zeta0)
    print('autoscale done')
    fig, ax = plt.subplots(figsize=(14,4))

    for index, L in enumerate(l):
        ax.plot(ws, a[index], label=L)
    ax.legend()
    ax.set_xlabel('Disorder strength', fontsize=13)
    ax.set_ylabel('Intrinsic dimension', fontsize=13)
    axin = ax.inset_axes([0.5, 0.5, 0.45, 0.45])

    #print(res)
    scaled = fssa.scaledata(l=l, rho=ws, a=a, da=da, rho_c=res['rho'], nu=res['nu'], zeta=res['zeta'])
    print('Scale data done')
    X = scaled[0]
    Y = scaled[1]
    #plt.figure()
    for index, L in enumerate(l):
        axin.plot(X[index], Y[index])

    quality = fssa.quality(X,Y,da)
    fig.suptitle('Mean ID with collapse on inset: w/ params: rho={}, nu={}, zeta={}'. format(round(res['rho'],2), 
                                                                                  round(res['nu'],2), 
                                                                                    round(res['zeta'],2)), fontsize=16)
    print('Quality check done') 

    plt.show()


def make_problem_sketch(num_points = 1000, elev=30, azim=65):
    x, y = np.random.randn(num_points), np.random.randn(num_points)
    z = x
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z, c=y, s=35)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(30, 65)

def eigenC_analysis(num_lims=8,L=8,min_disorder =0.5, max_disorder=5.5 ,steps=11, seeds=10, location='data/'):
    ws=np.linspace(min_disorder, max_disorder, steps)
    lims=np.logspace((1-num_lims),0,num_lims)
    maxs, lower_than = np.zeros((steps, seeds)), np.zeros((steps, num_lims, seeds))
    for index0, W in enumerate(ws):
        for index1, seed in enumerate(range(seeds)):
            filename = location+'/results-L-{}-W-{}-seed-{}.npz'.format(L, W, seed)
            eigs = abs(load_eigs(filename, 'vecs').flatten())
            maxs[index0,index1] = np.max(eigs)
            for index2, lim in enumerate(lims):
                count = count_lower_than(eigs, lim)
                lower_than[index0,index2, index1] = count
    means = np.mean(lower_than, axis=2).T/binomial(L)**2
    return means, maxs, lims

def eigenC_plots(means, maxs, 
                 lims = np.logspace((1-8),0,8),
                 min_disorder =0.5, max_disorder=5.5 ,steps=11, seeds =10, L=8,
                 colors = 'orange, lightblue, salmon, yellowgreen, grey, purple'.split(', ')
                ):
    
    ws=np.linspace(min_disorder, max_disorder, steps)
    # Plot 1: proportion below threshhold

    fig, ax  = plt.subplots(2,1, sharex=True, 
                           gridspec_kw={'height_ratios':[2,1]},
                           figsize=(10,6))
    #fig,ax = plt.subplots()
    #ax2=ax.twinx()
    for i, color in zip(range(len(means)), colors):

        ax[0].fill_between(ws, means[i], means[i+1],
                         label=lims[i],
                         color=color, alpha=.3)
    ax[0].legend(bbox_to_anchor=(1, 1.), )

    #plt.xlabel('Disorder strength, $W$', fontsize=12)
    ax[0].set_ylabel('Proportion of $|\lambda_c|<\zeta$ ', fontsize=12)
    #plt.title('Eigencomponent, $\lambda_c$, Dominance-chart', fontsize=14)

    #plt.savefig('figures/Domination-chart-L{}-seeds{},ws{}.png'.format(L,seeds,len(ws)), dpi=500, bbox_inches='tight')
    
    
    # Plot 2: Maxs
    #plt.figure()
    for index, i in enumerate(maxs):
        ax[1].scatter([ws[index]]*seeds, 1-i, c='b', alpha=2/seeds)
        ax[1].scatter([ws[index]], 1-np.mean(i), c='r', alpha=0.9)
        
    ax[1].legend(["point", "mean"],#bbox_to_anchor=(0.2, .25),
    	facecolor='white', framealpha=1)

    plt.xlabel('Disorder strength, $W$', fontsize=13)
    ax[1].set_ylabel('$1-max(|\lambda_c|)$', fontsize=13)
    #plt.title('Dominance of largest eigencomponent, $\lambda_c$', fontsize=14)
    #.grid()
    
    plt.suptitle('Eigencomponent, $\kappa$, dominance', fontsize=16)

    plt.savefig('figures/Domination-chart_comb-L{}-seeds{},ws{}.png'.format(L,seeds,len(ws)), dpi=500, bbox_inches='tight')
    
def get_slope_loss_and_weight(ws, seeds, L, location='data/'):
    

    slope_loss_and_weight = np.zeros((len(ws),seeds,3))
    index0 =-1
    for W in tqdm(ws):
        index0 += 1
        for index1, seed in enumerate(range(seeds)):
            filename = location+'/results-L-{}-W-{}-seed-{}.npz'.format(L, W, seed)
            eigs = load_eigs(filename, 'vecs')
            x,y,slope = nn2(eigs, plot=False, return_xy=True)
            loss = L2_loss_linear_originBound(x,y,slope)
            weight = weigt_from_loss(loss)
            slope_loss_and_weight[index0,index1] = np.array([slope,loss,weight])
    return slope_loss_and_weight

def plot_ID_weights(slope_loss_and_weight, L, seeds, ws):
    fig, ax = plt.subplots(1,2, figsize=(8,4), sharey=True)
    ax[0].set_ylabel('Disorder Strength', fontsize=13)
    pos0 = ax[0].imshow(slope_loss_and_weight[:,:,0], aspect=8, cmap='viridis') # disorder on x-axis, seed on y-axis, weight & slope by color or number
    pos1 = ax[1].imshow(slope_loss_and_weight[:,:,2], aspect=12, cmap='magma_r') # disorder on x-axis, seed on y-axis, weight & slope by color or number
    fig.suptitle('Intrinsic dim. and weight', fontsize=16)
    fig.text(0.3,0.075,'Disorder realization (seed)', fontsize=13)
    fig.colorbar(pos1, ax=ax[1], )
    fig.colorbar(pos0, ax=ax[1])
    plt.savefig('figures/ID-and-weights-L{}-seeds{},ws{}.png'.format(L,seeds,len(ws)), dpi=500, bbox_inches='tight')


def weighted_average_m1(distribution, weights): 
    # https://towardsdatascience.com/3-ways-to-compute-a-weighted-average-in-python-4e066de7a719
    numerator = sum([distribution[i]*weights[i] for i in range(len(distribution))])
    denominator = sum(weights)
    
    return numerator/denominator


def distance_between_vectors_euclidean_dotProduct(a,b):
    return np.sqrt(np.dot(a-b,a-b))


def nn2_new(A):
    N  = len(A)
    #Make distance matrix
    dist_M = np.array([[distance_between_vectors_euclidean_dotProduct(a,b) if index0 < index1 else 0 for index1, b in enumerate(A)] for index0, a in enumerate(A)])
    dist_M += dist_M.T + np.eye(N)*42
    
    # Calculate mu
    argsorted = np.sort(dist_M, axis=1)
    mu =  argsorted[:,1]/argsorted[:,0]
    x = np.log(mu)
    
    # Permutation
    y = np.array([1-dict(zip(np.argsort(mu)+1,(np.arange(1,N+1)/N)))[i+1] for i in range(N)])
    
    x,y  = x[y>0], y[y>0]
    y = -1*np.log(y)
    
    #fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
    
    # Goodness
    _, pvalue = chisquare(f_obs=x*d , f_exp=y, ddof=1)
    
    return d, pvalue
