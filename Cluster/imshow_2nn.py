import numpy as np
import matplotlib.pyplot as plt

path = "/home/projects/ku_00067/scratch/mbl-intrinsicdimension/results_NEW/"

L = 12

data = []

for seed in range(100):
	data_1_file = np.load(path+'{}-{}.npy'.format(L, seed))
	data.append(data_1_file)


data = np.array(data).T
print(data)
plt.imshow(data)
plt.savefig("nn2_output0.png", dpi=500)
