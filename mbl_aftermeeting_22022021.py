'''
Author = Anton Golles
'''

from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, spmatrix, linalg, save_npz, lil_matrix

def binary_conv(x, L):
	'''
	convert base-10 integer to binary and adds zeros to match length
	returns: Binary 
	'''
	b = bin(x).split('b')[1]
	while len(b) < L:
		b = '0'+b
	return b

def count_ones(bitstring):
	'''
	Takes binary bitstring and counts number of ones
	returns: number of ones
	'''
	counta = 0
	for i in bitstring:
		if i=='1':
			counta += 1
	return counta

def binomial(n, k='half'):
	'''
	find binomial coefficient of n pick k,
	returns interger
	'''
	if k=='half':
		k = n//2
	return int(factorial(n)/factorial(k)**2)

def energy_diag(bitString, V, U):
	'''
		Determines the energy for the diagonal of the Hamiltonian
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

def basis_states(L, W, U, t):
	'''
		Look for basis states
		Returns dictionaries (State_to_index,index_to_State)
	'''
	if L%2!=0:
		print('Please input even int for L')

	s2i = {} # State_to_index
	i2s = {} # index_to_State

	index = 0
	for i in range(int(2**L)): # We could insert a minimum
		binary = binary_conv(i, L)
		ones = count_ones(binary)

		if ones == L//2:
			s2i[binary] = index
			i2s[i] = binary
			index +=1

	return (s2i, i2s)

def construct_Hamiltonian(L = 4, W = 2, U = 1, t = .42, seed=42):
	'''
	Constructs the sparse Hamiltonian matrix
	returns: Hamiltonian
	'''
	np.random.seed(seed)
	V = np.random.uniform(-1,1,size=L) * W
	num_states = binomial(L)

	(s2i, i2s) = basis_states(L, W, U , t)

	H = lil_matrix((num_states,num_states))

	for key in s2i.keys():
		
		E = energy_diag(key, V, U)
		H[s2i[key],s2i[key]] = E
		for site in range(L):
			
			try:
				if (key[site] == '1' and key[site+1]== '0'):
					
					new_state = key[:site] + '0' + '1' + key[site+2:]
					#print(key,site, new_state, 'nn')
					H[s2i[key],s2i[new_state]] = t
					H[s2i[new_state],s2i[key]] = t

			except IndexError: # periodic boundary conditions
				if (key[site] == '1' and key[0]== '0'):
					
					new_state = '1' + key[1:site] + '0'
					#print(key,site, new_state, 'yy')
					H[s2i[key],s2i[new_state]] = t
					H[s2i[new_state],s2i[key]] = t
	return H

def diag_sparse(sparse_hamiltonian, k=3):
    (eigvals, eigvecs) =  linalg.eigsh(sparse_hamiltonian, k=k)
    return (eigvals, eigvecs.T)

def level_spacing(eigvals):
	level_spacing = [abs(eigvals[i]-eigvals[i+1]) for i in range(len(eigvals)-1)]
	level_spacing.append(abs(eigvals[len(eigvals)-1]-eigvals[0]))
	return np.mean(level_spacing)

print(diag_sparse(construct_sparse_Hamiltonian()))

'''
Ws = np.logspace(0,1,40)
mean_level_spacings = []
for W in Ws:
	mls = construct_Hamiltonian(L,s2i,i2s, W, U, t)
	mean_level_spacings.append(mls)
print(Ws)
print(mean_level_spacings)


plt.figure(figsize=(12,6))
plt.scatter(Ws, mean_level_spacings)
plt.grid()
plt.savefig('../images/mean_level_spacings.png',dpi=300)

'''