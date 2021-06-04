
import numpy as np
from utils import nn2



def nn2_new(L=8, first_seed=0):
        vals_or_vecs = 'eigvecs'

        Ws = np.linspace(0.1,6.1,31)
        last_seed = first_seed + 99

        filename = '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-{}/eigvecs-and-eigvals-L-{}-seeds-{}-{}.npy'.format(L,L,first_seed, last_seed)
        data = np.load(filename, allow_pickle=True)

        seeds = np.arange(first_seed, last_seed)
        D = np.zeros((99,len(Ws)))
        for index1, seed in enumerate(seeds):
                for index2, W in enumerate(Ws):
                        W = round(W,1)
                        eigs = data.item()[seed][W][vals_or_vecs]
                        d = nn2(eigs)
                        D[index1,index2] = d
        return D
