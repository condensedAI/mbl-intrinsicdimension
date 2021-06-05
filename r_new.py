import numpy as np
from utils import level_spaces, r_stat

def R(L=10,seeds=10000, single=False):

	Ws = np.linspace(0.1,6.1,31)

	R_main = []
	R_std_main = []
	if single == False:
		for first_seed in np.arange(0,seeds,100):
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
			R_std_main.append(np.mean(R_std**2,axis=0)**.5)	
	
	
		R, std = np.array(R_main), np.array(R_std_main)
		r = np.mean(R, axis=0)
		std = np.mean(std**2, axis=0)**.5



	else:
		for seed in range(seeds):
			try:
				filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-seed-{}.npy'.format(L,L,seed)
				data = np.load(filename, allow_pickle=True)
				R = np.zeros(len(Ws))
				R_std = np.zeros(len(Ws))
				for index2, W in enumerate(Ws):
					W = round(W,1)
					eigs = data.item()[W][vals_or_vecs]
					r = r_stat(eigs)
					R[index2] = np.mean(r)
					R_std[index2] = np.std(r)
				R_main.append(np.mean(R,axis=0))
				R_std_main.append(np.mean(R_std**2,axis=0)**.5)
			except:
				pass

		R, std = np.array(R_main), np.array(R_std_main)
		r = np.mean(R, axis=0)
		std = np.mean(std**2, axis=0)**.5
	np.savez('fullresults/R_stat_L{}.npz'.format(L), [r,std])					
