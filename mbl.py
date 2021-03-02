'''
Author = Anton Golles
'''

from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import diags, spmatrix, linalg, save_npz, lil_matrix, csr_matrix
from datetime import datetime
from tqdm import tqdm
import os

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

def benchmark(num_seeds=200, Lmax = 12):
	'''
	benchmarks build and diag of hamiltonian with Dense vs. Sparse matricies. Plots average run time with std
	______________
	Parameters:
		num_seeds: number of realizations; int
		Lmax: max size of system; int
	Returns:
		its a procedure
	'''
	times =[[],[]]
	Ls = np.arange(2,Lmax,2)
	methods, color = ['sparse', 'dense'], ['b', 'g']

	for L in Ls:
		print('###########\n\n', L)
		times_sparse = []
		times_dense = []
		for seed in range(20, num_seeds+20):
			print(seed)
			for method in methods:
				startTime = datetime.now()
				diag(constructHamiltonian(L=L, seed=seed, method=method))
				time = (datetime.now() - startTime).total_seconds()
				if method == 'sparse':
					times_sparse.append(time)
				else:
					times_dense.append(time)

		times[0].append(times_sparse)
		times[1].append(times_dense)

	
	for index, t, in enumerate(times):
		time, std = np.mean(t, axis=1), np.std(t, axis=1)

		plt.plot(Ls, time, label = methods[index], c=color[index])
		[plt.plot(Ls, i,c='orange', ls='--') for i in [time+std, time-std]]
	
	plt.legend()
	plt.xlabel('L')
	plt.ylabel('time to run (s)')
	plt.yscale('log')
	plt.grid()
	plt.title('Benchmark run time, build+diag Hamiltonian, realizations = {}'.format(num_seeds))
	plt.savefig('../images/benchmarks.png', dpi=300, transparent=True)

def arrToString(arr):
	a = str(arr).replace('[','').replace(']','').replace(' ',',').replace('\n',',').replace(',,', ',')
	return a

def buildnDiag(
	method = 'dense',
	L_high = 12,
	disorder_realizations = 10,
	disorders = 10,
	):
	for L in np.arange(8, L_high, 2): # System sizes
		print('L =', L)
		for w in tqdm(np.linspace(1,5,disorders)):
			for seed in range(disorder_realizations):
				H = constructHamiltonian(L = L, W = w, seed=seed, method=method)
				eigvals, eigvecs = diag(H)
				np.savez('results1/results-L-{}-W-{}-seed-{}.npz'.format(L, w, seed) ,eigvals, eigvecs)



def load_eigs_npz(filename='results/results-L-2-W-0.1-seed-0.npz'):
	data = np.load(filename)
	eigvals = data[data.files[0]]
	eigvecs = data[data.files[1]]

	return eigvals, eigvecs

def load_eigvals_npz(filename='results1/results-L-2-W-0.1-seed-0.npz'):
	data = np.load(filename)
	eigvals = data[data.files[0]]
	return eigvals

def load_eigvecs_npz(filename='results/results-L-2-W-0.1-seed-0.npz'):
	data = np.load(filename)
	eigvecs = data[data.files[1]]

	return eigvecs

def levelSpacing(eigvals):
	'''
	given eigenvalues, finds level spacing
	find gap between elements in a list
	periodic boundary conditions
	______________
	Parameters:
		eigenvals : list of floats
	______________
	Returns:
		Level spacings : list of floats of len(eigvals)
	'''
	s = [eigvals[i+1]-eigvals[i] for i in range(len(eigvals)-1)]
	s.append(abs(eigvals[len(eigvals)-1]-eigvals[0]))  # Periodic boundary
	return s


def rStatFromFiles(
	L_high = 6,
	disorder_realizations = 10,
	disorders = 10):
	rs = []
	for L in np.arange(4, L_high, 2): # System sizes
		print(L)
		r = []
		for w in np.linspace(0.3,6,disorders):
			if w < 1:
				pass
			else:
				r_temp = []
				for seed in range(disorder_realizations):
					filename = os.path.expanduser("~/results1/results-L-{}-W-{}-seed-{}.npz".format(L, w, seed))
					eigvals = load_eigvals_npz(filename)
					level_spacings = levelSpacing(eigvals)
					r_temp_temp = 0
					length = len(level_spacings)
					for i in range(length-1):
						r_temp_temp += min(level_spacings[i:i+2])/max(level_spacings[i:i+2])
					r_temp.append(r_temp_temp/length)
				print(filename)
				r.append(np.mean(r_temp))
		rs.append(r)
	return rs

def rStatFromFiles_centerEigs(
	L_high = 6,
	disorder_realizations = 10,
	disorders = 10,
	location = '~/results1/'):
	rs = []
	for L in np.arange(4, L_high, 2): # System sizes
		print(L)
		r = []
		for w in np.linspace(0.3,6,disorders):
			if w < 1:
				pass
			else:
				r_temp = []
				for seed in range(disorder_realizations):
					filename = os.path.expanduser("~/results1/results-L-{}-W-{}-seed-{}.npz".format(L, w, seed))
					eigvals = load_eigvals_npz(filename)
					level_spacings = levelSpacing(eigvals)
					r_temp_temp = 0
					length = len(level_spacings)
					for i in range(int((length-1)//4),int(3*(length-1)//4)):
						r_temp_temp += min(level_spacings[i:i+2])/max(level_spacings[i:i+2])
					r_temp.append((r_temp_temp)/(length/2))
				print(filename)
				r.append(np.mean(r_temp))
		rs.append(r)
	return rs



def plotR(rs, L_high=14, disorders=17,name='2', title='eigs'):
	for index, L in enumerate(np.arange(4,L_high,2)):
		if L < 8:
			pass
		else:
			plt.scatter(np.linspace(1.2,6,disorders), rs[index], label=L)

	plt.legend()
	plt.grid()
	plt.title('r-statistic for different system sizes, {}'.format(title))
	plt.xlabel('disorder strength, $w$')
	plt.ylabel('r-statistic')

	plt.savefig('r_test{}.png'.format(name),dpi=420)

