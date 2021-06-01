import numpy as np
from utils import level_spaces, r_stat

L=10

Ws = np.linspace(0.1,6.1,31)
print(Ws)

R_main = []
R_std_main = []

for first_seed in np.arange(0,10000,100):
	last_seed = first_seed + 99

	filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-seeds-{}-{}.npy'.format(L,L,first_seed, last_seed)
	data = np.load(filename, allow_pickle=True)

	seeds = np.arange(first_seed, last_seed)
	vals_or_vecs = 'eigvals'

	R = np.zeros((99,len(Ws)))
	R_std = np.zeros((99,len(Ws)))
	for index1, seed in enumerate(seeds):
		for index2, W in enumerate(Ws):
			W = round(W,1)
			eigs = data.item()[seed][W][vals_or_vecs]
			r = r_stat(eigs)
			R[index1,index2] = np.mean(r)
			R_std[index1,index2] = np.std(r)


	R_main.append(np.mean(R,axis=0))
	R_std_main.append(np.mean(R_std,axis=0))	
	
	
R, std = np.array(R_main), np.array(R_std_main)
#print(np.shape(R))
#print(np.shape(std))
#print(R)
print(np.mean(R, axis=0))
print(np.mean(std, axis=0))
