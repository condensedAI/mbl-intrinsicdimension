from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def pca(data, n_components=20):
    data_scaled = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data_scaled)
    return pca.explained_variance_ratio_
def load_eigs_npz(filename='results/results-L-2-W-0.1-seed-0.npz', what='vals'):
    data = np.load(filename)
    if what == 'vals':
        return data[data.files[0]]
    else:
        return data[data.files[1]].flatten()
    
def file_path(L,w,seed):
    return '/home/projects/ku_00067/scratch/mbl-intrinsicdimension/data/results-L-{}-W-{}-seed-{}.npz'.format(L,w,seed)  


Ls = [10,12]
ws = [1.0, 1.55, 2.09, 2.64, 3.18, 3.73, 4.27, 4.82, 5.36, 5.91, 6.45, 7.0]
seeds = np.arange(0,60,1)

exp_var_many = np.zeros((len(Ls),len(ws)))
for index0, L in enumerate(Ls):
    for index1, w in enumerate(tqdm(ws)):
        X = np.array([load_eigs_npz(file_path(L, w,seed), what='vecs') for seed in seeds])
        exp_var = pca(X, n_components=L)
        exp_var_many[index0,index1] = sum(exp_var)

np.savez('exp_var_many.npz', exp_var_many)