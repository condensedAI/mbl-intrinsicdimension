
import numpy as np
from utils import nn2



def nn2_new(L=8, first_seed=0):

	Ws = np.linspace(0.1,6.1,31)
	last_seed = first_seed + 9

	if L in [8,10]:
		filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-seeds-{}-{}.npy'.format(L,L,first_seed, last_seed)
		data = np.load(filename, allow_pickle=True)
		seeds = np.arange(first_seed, last_seed)
		D = np.zeros((99,len(Ws)))
		for index1, seed in enumerate(seeds):
			for index2, W in enumerate(Ws):
				W = round(W,1)
				eigs = data.item()[seed][W]['eigvecs']
				d = nn2(eigs)
				D[index1,index2] = d
	elif L in [12,14]
		filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-seed-{}.npy'.format(L,L,first_seed, last_seed)
		data = np.load(filename, allow_pickle=True)
		D = np.zeros(len(Ws))
		seed = first_seed
		for index2, W in enumerate(Ws):
			W = round(W,1)
			eigs = data.item()[W]['eigvecs']
			d = nn2(eigs)
			D[index2] = d
	elif L in [16]:
		D = np.zeros(len(Ws))
		for index, W in enumerate(Ws):
			filename='/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-W-{}-seed-{}.npy'.format(L,L,W,first_seed)
			W = round(W,1)
			data = np.load(filename, allow_pickle=True)
			eigs = data.item()[W]['eigvecs']
			D[index] = nn2(eigs)
	return D
