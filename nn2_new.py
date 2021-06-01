
import numpy as np
from utils import nn2

L=8
vals_or_vecs = 'eigvecs'

Ws = np.linspace(0.1,6.1,31)
print(Ws)

D_main = []
D_std_main = []

for first_seed in np.arange(0,10000,100):
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

        D_main.append(np.mean(D,axis=0))
        D_std_main.append(np.std(D_std,axis=0))


D, std = np.array(D_main), np.array(D_std_main)
#print(np.shape(D))
#print(np.shape(std))
#print(D)
print(np.mean(D, axis=0))
print(np.mean(std, axis=0))

