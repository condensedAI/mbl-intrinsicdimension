'''
Author = Anton Golles
'''

from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import diags, spmatrix, linalg, save_npz, lil_matrix
from datetime import datetime

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
	'''
	Diagonalizes sparse Hamiltonian
	Returns (eigenvalues, eigenvectors)
	'''
    (eigvals, eigvecs) =  linalg.eigsh(sparse_hamiltonian, k=k)
    #print(eigvals)
    return (eigvals, eigvecs.T)

def level_spacing(eigvals):
	s = [eigvals[i+1]-eigvals[i] for i in range(len(eigvals)-1)]
	s.append(abs(eigvals[len(eigvals)-1]-eigvals[0]))
	return s


def obtain_s_r(num_w=4, num_seeds=10,L=10,save=False):

	s = []
	ws = np.linspace(.2,6,num_w)
	for w in ws:
		print('W = ',w)
		S = []
		for seed in range(num_seeds):
			(eigvals, eigvecs) = diag_sparse(construct_Hamiltonian(L = L, W = w, seed=seed))
			ss = level_spacing(eigvals)
			S.append(ss)
		S = list(np.array(S).flatten())
		s.append(S)
	rs = []

	for index,i in enumerate(s):
		r= [min(i[j:j+2])/max(i[j:j+2]) for j in range(len(i)-1)]
		rs.append(r)
	if save==True:
		np.savetxt('s{}.csv'.format(L), s, delimiter=',')
		np.savetxt('r{}.csv'.format(L), rs, delimiter=',')

	return (s,rs)


def make_hist(s,ws):
	colors = 'red, green, blue, orange, black'.split(', ')*8
	plt.figure()
	for index,i in enumerate(s):
		sns.histplot(i, 
			kde=False, color=colors[index], binwidth=.3,
			stat="probability",
			fill=False,  thresh=.3,
			label='w={}'.format(ws[index]))
	plt.title('P(s)')
	plt.xlabel('level spacings, $s$')
	plt.legend()
	plt.savefig('../images/hist.png', dpi=300)


def make_kde(s,ws):
	colors = 'red, green, blue, orange, black'.split(', ')*9
	plt.figure()
	for index,i in enumerate(s):
		sns.kdeplot(i, 
			color=colors[index],
			label='w={}'.format(ws[index]))
	plt.title('P(s)')
	plt.xlabel('level spacings, $s$')
	plt.legend()
	plt.savefig('../images/kde.png', dpi=300)

def plot_r(ws,rs):
	#plt.figure()
	plt.plot(ws,[np.mean(r) for r in rs])
	plt.ylabel('r')
	plt.xlabel('w')
	#plt.savefig('../images/r.png', dpi=300)	

def load_r(filename):
	r = np.loadtxt(filename, delimiter=',')
	return r


def time_to_run(Ls = [4,6,8,10,12,14], num_w=1, seeds=100):
	data = ['L,ws,seeds,time'.split(',')]
	for L in Ls:
		startTime = datetime.now()
		obtain_s_r(num_w,seeds,L)
		time = (datetime.now() - startTime).total_seconds()
		time = str(time)
		#time = float(time)
		data.append([L,num_w,seeds, time])
	data = np.array(data)

	np.savetxt('time_to_run.csv', data, delimiter=',')
	print(data)



num_w = 10
ws = np.linspace(0.1,6,num_w)

#for L in [8,10,12]:
	#(s,r) = obtain_s_r(num_w=num_w,num_seeds=1000, L=L,save=True)

#np.savetxt('ws.csv', ws, delimiter=',')



#time_to_run()


'''
plot_r(ws,load_r('r8.csv'))
plot_r(ws,load_r('r10.csv'))
plot_r(ws,load_r('r12.csv'))
plt.legend(['L=8','L=10','L=12'])
plt.savefig('../images/rs.png', dpi=300)

'''


