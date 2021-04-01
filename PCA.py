import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


def pca(data, n_components=2):
    data_scaled = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data_scaled)
    #print('pca.explained_variance_ratio_',
    #      pca.explained_variance_ratio_, 'sum = ',
    #return sum(pca.explained_variance_ratio_)#, principalComponents
    return principalComponents,sum(pca.explained_variance_ratio_)

    
def load_eigs_npz(filename='results/results-L-2-W-0.1-seed-0.npz'):
    data = np.load(filename)
    eigvals = data[data.files[0]]
    eigvecs = data[data.files[1]]

    return eigvals, eigvecs.T

def PCA_for_many():
	L = int(input('L: '))
	seeds = int(input("Number of seeds (disorder realizations): "))
	print("disorder low, high, steps")
	low = int(input("low: "))
	high = int(input("high: "))
	steps = int(input("steps:"))


	plt.figure(figsize=(8,8))

	Files = []


	ws = np.linspace(low,high,steps)
	seeds = np.arange(0,seeds,1)

	for w in ws:
		files = []
		for seed in seeds:
			file = 'data/results-L-{}-W-{}-seed-{}.npz'.format(L,round(w,2),seed)
			files.append(file)
		Files.append(files)


	eigs = []
	for files in Files:
		dd = []
		for file in tqdm(files):
			eigvals, eigvecs = load_eigs_npz(file)
			data=eigvecs.flatten()
			dd.append(data)
		eigs.append(dd)


	eigs = np.array(eigs)

	for num_components in [1,2]:#,5,10,15,25,40,60,100]:
		exps = []
		for i in eigs:
			print(np.shape(i.T))
			pc, exp = pca(i, num_components)
			exps.append(exp)

		plt.scatter(ws,exps, color=colors[index], s=100, label=LL)

		plt.title('Variance in flattend eigenvectors explain by {} PC'.format(num_components))
		plt.xlabel('Disorder strength, $W$')
		plt.ylabel('Explained variance')
		plt.legend()
		plt.savefig('pca{}PC.png'.format(num_components))


PCA_for_many()

