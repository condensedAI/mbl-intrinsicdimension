import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm
from scipy.sparse import lil_matrix, linalg
from sklearn.cluster import AffinityPropagation, Birch, SpectralClustering, MeanShift
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, KMeans
import seaborn as sns

def plot_form(xlabel='X', 
	ylabel='Y',title='title',legend=True, ax=plt):
    if legend == True:
        plt.legend(fontsize=12)
    ax.title(title, fontsize=12)
    ax.xlabel(xlabel, fontsize=12)
    ax.ylabel(ylabel, fontsize=12)
    
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

def buildDiagSave(method = 'dense'):
    '''
    builds Hamiltonians, diagonalizes and saves eigvecs and eigvals    
    '''
    #print('L =', L)
    L = int(input('L: '))
    seeds = int(input("Number of seeds (disorder realizations): "))
    print("disorder low, high, steps")
    low = int(input("low: "))
    high = int(input("high: "))
    steps = int(input("steps:"))
    
    
    for w in tqdm(np.linspace(low,high,steps)):
        for seed in range(seeds):
            H = constructHamiltonian(L = L, W = w, seed=seed, method=method)
            eigvals, eigvecs = diag(H)
            np.savez('data/results-L-{}-W-{}-seed-{}.npz'.format(L, round(w,2), seed, method) ,eigvals, eigvecs)

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

## R-statistic
def level_spaces(lst):
    return [abs(lst[i]-lst[i+1]) for i in range(len(lst)-1)] + [lst[len(lst)-1]- lst[0]]

def r_stat(lst):
    spaces = level_spaces(lst)
    return [min(spaces[i:i+2])/max(spaces[i:i+2]) for i in range(len(spaces))]

def r_stat_many(eigs):
    r = [[[r_stat(eigs) for eigs in eigs_w]for eigs_w in eigs_L] for eigs_L in eigs]
    r = np.array(r)
    r_shape = np.shape(r)
    r = r.reshape(r_shape[0],r_shape[1],r_shape[2]*r_shape[3])
    return np.mean(r, axis=2), np.std(r, axis=2)


## Clustering
def run_clustering(X, which=['KMeans', 'OPTICS', 'meanshift']):
    #print(np.shape(X))
    X = X.reshape(np.shape(X)[1]* np.shape(X)[2], np.shape(X)[3], order='F')
    #print(np.shape(X))
    results = []
    if which == 'all':
        which=['KMeans', 'Affinity Propagation', 'Birch', 'Spectral Clustering', 
               'Agglomerative Clustering', 'DBSCAN', 'MeanShift', 'OPTICS']
    if 'KMeans' in which:
        kmeans = KMeans(2, verbose=0).fit(abs(np.array(X)))
        results.append(kmeans.labels_)
    if 'Affinity Propagation' in which:
        affinityPropagation = AffinityPropagation(random_state=5, damping=.99).fit(X)
        results.append(affinityPropagation.labels_)
    if 'Birch' in which:
        brc = Birch(n_clusters=2)
        brc.fit(X)
        birch_labels = brc.predict(X)
        results.append(birch_labels)
    if 'Spectral Clustering' in which:
        spectralClustering = SpectralClustering(n_clusters=2,
                                                assign_labels="discretize",
                                                random_state=0).fit(X)
        results.append(spectralClustering.labels_)
    if 'Agglomerative Clustering' in which:
        agglomerativeClustering = AgglomerativeClustering().fit(X)
        results.append(agglomerativeClustering.labels_)
    if 'DBSCAN' in which:
        db = DBSCAN(eps=3, min_samples=8,leaf_size=10).fit(X)
        results.append(db.labels_)
    if 'MeanShift' in which:
        meanShift = MeanShift(bandwidth=20).fit(X)
        results.append(meanShift.labels_)
    if 'OPTICS' in which:
        optics = OPTICS(min_samples=2).fit(X)
        results.append(optics.labels_)
        
    return np.array(results), which

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
    
    indices = np.argsort(x)
    print(indices)
    print(indices[-del_vals:])
    x = x[indices[:-del_vals]]
    y = y[indices[:-del_vals]]
    x2 = x[indices[-del_vals:]]
    y2 = y[indices[-del_vals:]]
	

    y = -1*np.log(y)
    y2 = -1*np.log(y2)
	
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
        return x,y, d,x2,y2
    else:
        return d