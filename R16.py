import numpy as np
from utils import r_stat

def R16(L=16, seed=0):
	Ws = np.linspace(0.1,6.1,31)
	R = {}
	for W in Ws:
		filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-W-{}-seed-{}.npy'.format(L,L,W,seed)
		W = round(W,1)
		try:
			data = np.load(filename, allow_pickle=True)
			print(data.item().keys())
			eigs = data.item()[W]['eigvals']
			R[W] = np.mean(r_stat(eigs))
		except:
			R[W]=None
	return R
