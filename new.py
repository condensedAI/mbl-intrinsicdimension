import numpy as np
from utils import level_spaces, r_stat

for first_seed in np.arange(0,500,100):
	last_seed = first_seed + 99

	filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-8/eigvecs-and-eigvals-L-8-seeds-{}-{}.npy'.format(first_seed, last_seed)
	data = np.load(filename, allow_pickle=True)

	first_seed, last_seed = int(filename.split('-')[8]), int(filename.split('-')[9].split('.')[0])

	seeds = np.arange(first_seed, last_seed)
	W = 0.1
	vals_or_vecs = 'eigvals'

	R = [[],[]] # mean and std
	for seed in seeds:
		eigs = data.item()[seed][W][vals_or_vecs]
		r = r_stat(eigs)
		R[0].append(np.mean(r))
		R[1].append(np.std(r))


	print(np.mean(R, axis=1))

