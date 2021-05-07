import numpy as np
import matplotlib.pyplot as plt

path = "/home/projects/ku_00067/scratch/mbl-intrinsicdimension/results_NEW/"

L = 12

data = []

for seed in range(1000):
	data_1_file = np.load(path+'{}-{}.npy'.format(L, seed))
	data.append(data_1_file)


data = np.array(data).T
print(data)

plt.figure(figsize=(10,2))
plt.imshow(data)
plt.xlabel('Seed number')
plt.ylabel("disorder")
plt.yticks(range(12),[1.0, 1.55, 2.09, 2.64, 3.18, 3.73, 4.27, 4.82, 5.36, 5.91, 6.45, 7.0])
plt.title("nn2_output-L-{}-seeds-{}_1.png".format(L, seed+1))
plt.savefig("nn2_output-L-{}-seeds-{}.png".format(L, seed+1), dpi=500)


x = [1.0, 1.55, 2.09, 2.64, 3.18, 3.73, 4.27, 4.82, 5.36, 5.91, 6.45, 7.0]
y = np.mean(data, axis=1)
std = np.std(data, axis=1)
plt.figure(figsize=(12,6))
plt.plot(x,y, label='L='.format(L), c='r')
plt.scatter(x,y, c='r')
plt.plot(x,y+std, ls='--', c='orange')
plt.plot(x,y-std, ls='--', c='orange')

plt.title('Mean result from 2nn')
plt.xlabel('disorder strength, $W$')
plt.ylabel('Intrinsic dimension')


plt.savefig('2nn_mean.png', dpi=500)
