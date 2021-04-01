import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm

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

def buildnDiag(method = 'dense'):
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
            np.savez('data/results-L-{}-W-{}-seed-{}.npz'.format(L, round(w,2), seed) ,eigvals, eigvecs)

buildnDiag()