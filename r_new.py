import numpy as np
from utils import level_spaces, r_stat

def R(L=10,seeds=100, first_seed=0):
	Ws = np.linspace(0.1,6.1,31)
	R = {}
	for W in Ws:
			W = round(W,1)
			R[W] = {}

	if L < 11:
		for first_seed in np.arange(0,seeds,100): # step Truough seeds
			try:
				last_seed = first_seed + 99

				filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-seeds-{}-{}.npy'.format(L,L,first_seed, last_seed)
				data = np.load(filename, allow_pickle=True)  # Contains 99*31*2 entries
				for index1, seed in enumerate(np.arange(first_seed, last_seed)):
					for index2, W in enumerate(Ws):
						eigs = data.item()[seed][round(W,1)]['eigvals']
						R[round(W,1)][seed] = r_stat(eigs)
			except:
				print('file could not be loaded:', filename)
				pass

	elif L < 15:
		for seed in np.arange(first_seed,first_seed+seeds,1):
			try:
				filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-seed-{}.npy'.format(L,L,seed)
				data = np.load(filename, allow_pickle=True)
				for index2, W in enumerate(Ws):
					W = round(W,1)
					eigs = data.item()[W]['eigvals']
					R[round(W,1)][seed] = r_stat(eigs)
			except:
				print('Not found:',filename)
	else:

		for W in Ws:
			filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-W-{}-seed-{}.npy'.format(L,L,W,first_seed)
			W = round(W,1)
			try:
				data = np.load(filename, allow_pickle=True)
				print(data.item().keys())
				eigs = data.item()[W]['eigvals']
				R[W] = np.mean(r_stat(eigs))
			except:
				R[W]=None
		return R

	print(R)
	R_means = {}
	for W in Ws:
		W = round(W,1)
		R_means[W] = np.mean(list(R[W].values()))
	print(R_means)
	return R_means
