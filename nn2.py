import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm
from scipy.spatial import distance_matrix
import time
from numba import jit

@jit(nopython=True)


def load_eigs_npz(filename='data/results-L-12-W-1.0-seed-42.npz'):
	data = np.load(filename)
	eigvals = data[data.files[0]]
	eigvecs = data[data.files[1]]
	return eigvals, eigvecs.T


def listFilenames(Ls = [8,10],
				  ws = [1.0, 1.44, 1.89, 2.33, 2.78, 3.22, 3.67, 4.11, 4.56, 5.0, 
						  5.56, 6.11, 6.67, 7.22, 7.78, 8.33, 8.89, 9.44],
				   seeds = np.arange(0,20,1)):
	print('number of disorders',len(ws))
	print('number of seeds',len(seeds))
	print('L', Ls[0])
	Files = []
	for L in Ls:
		Files_L = []
		for w in ws:
			w = round(w,2)
			files = []
			for seed in seeds:
				file = 'data/results-L-{}-W-{}-seed-{}.npz'.format(L,w,seed)
				files.append(file)
			Files_L.append(files)
		Files.append(Files_L)
	print('Shape Files',np.shape(Files))
	return Files, ws, Ls
	

def load_many_eigs(Files):  
	print(len(np.shape(Files)))
	
	if len(np.shape(Files)) == 3:
		eigs = []
		
		for Files_L in tqdm(Files):
			DD = []
			for files in Files_L:
				dd = []
				for file in files:
					eigvals, eigvecs = load_eigs_npz(file)
					data=eigvecs
					dd.append(data)
				DD.append(dd)
			eigs.append(DD)
		
	else:
		eigs = []

		for files in tqdm(Files):
			dd = []
			for file in files:
				eigvals, eigvecs = load_eigs_npz(file)
				data=eigvecs
				dd.append(data)
			eigs.append(dd)
	print('Eigs Loaded!')
	eigs = np.array(eigs, dtype='object')
	return eigs

def dist_Matrix(data):
	dist_matrix = np.zeros((N, N))
	# Making the distance matrix: distance from each eigvec to all others
	for i, eigvec1 in enumerate(data):
		for j, eigvec2 in enumerate(data):	
			distance = sum(abs((eigvec1-eigvec2)))
			dist_matrix[i,j], dist_matrix[j,i] = distance, distance
	return dist_matrix


def nn2(
	data,
	plot=False,
	return_xy = False,
	plot_index=0,
	plot_index1=0,
	w=1):
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
	starttime = time.time()
	dist_M = dist_Matrix(data)
	print("Old Distance matrix took %.5f seconds"%(time.time() - starttime))	
	#starttime = time.time()
	#dist_M = distance_matrix(data,data, p=1)
	#print("New dist took %.5f seconds"%(time.time() - starttime))
	# table of distances - state and \mu= r_2/r_1
	mu = np.zeros((N,2))
	for index, line in enumerate(dist_M):
		r1, r2 = sorted(line)[1:3]
		mu[index,0] = index+1
		mu[index,1] = r2/r1
		

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

	#fit line through origin to get the dimension
	d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]

	if plot==True:
		#fig, ax = plt.subplots(2,1, sharex=True)
		plt.scatter(x,y, c='g')
		plt.plot(x,x*d, c='r', ls='--')
		
	  #ax[plot_index].set_xlabel('log($\mu$)', fontsize=12)
		plt.text(0,5,'w={}'.format(w), fontsize=13)
		plt.text(0,4.5,'d={}'.format(round(d,3)), fontsize=13)
		if plot_index==0:
			ax[plot_index1, plot_index].set_ylabel('$1-F_i(\mu)$', fontsize=12)
			
	if return_xy:
		return x,y, d
	else:
		return d

def get_intrinsic_dims(eigs):
	intrinsic_dims = [] # for various W
	for eig_L in eigs:
		intrinsic_dim_L = []
		for eig in tqdm(eig_L):
			intrinsic_dim = []
			for data in tqdm(eig):
				d= nn2(data)
				intrinsic_dim.append(d)
			intrinsic_dim_L.append(intrinsic_dim)
		intrinsic_dims.append(intrinsic_dim_L)


	return intrinsic_dims



L = int(input('L: '))
num_seeds = int(input("Number of seeds (disorder realizations): "))
print("disorder low, high, steps")
low = int(input("low: "))
high = int(input("high: "))
steps = int(input("steps:"))

filenames, ws, Ls = listFilenames(Ls=[L],
						  ws = np.linspace(low,high,steps),
						 seeds = np.arange(num_seeds)
						 )
eigs = load_many_eigs(filenames)
IDs = get_intrinsic_dims(eigs)

#print(IDs)

np.savez('results/nn2_results_L{}_seeds{}_low{}_high_steps{}_new.npz'.format(L,num_seeds, low, high, steps), IDs, ws)
