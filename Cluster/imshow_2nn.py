import numpy as np
import matplotlib.pyplot as plt

path = "/home/projects/ku_00067/scratch/mbl-intrinsicdimension/results_NEW/"

L = 14

data = []

for seed in range(1000):
	try:
		data_1_file = np.load(path+'{}-{}.npy'.format(L, seed))
		data.append(data_1_file)
	except:
		pass


data = np.array(data).T
#print(data)



def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.imshow(data, extent=[0,1000,7,1])
forceAspect(ax,aspect=4)

fig.suptitle("2NN output;L={}, seeds={}_1".format(L, seed+1), fontsize=16)
fig.savefig("nn2_output_L_{}_seeds_{}.png".format(L, seed+1), dpi=500, bbox_inches='tight')
np.savez('2nn_14_1000.npz', data)



Ls = [12,14]
color1 = ['red', 'green']
color2 = ['salmon', 'lightgreen']

plt.figure(figsize=(12,6))
for L,c1,c2 in zip(Ls, color1, color2):
	data = []

	for seed in range(1000):
		try:
			data_1_file = np.load(path+'{}-{}.npy'.format(L, seed))
			data.append(data_1_file)
		except: 
			pass

	data = np.array(data).T
	print('L=',L, data[0])

	x = [1.0, 1.55, 2.09, 2.64, 3.18, 3.73, 4.27, 4.82, 5.36, 5.91, 6.45, 7.0]
	y = np.mean(data, axis=1)
	print(y)
	std = np.std(data, axis=1)
	plt.plot(x,y, label='L={}'.format(L), c=c1)
	plt.scatter(x,y, c=c1)
	plt.plot(x,y+std, ls='--', c=c2)
	plt.plot(x,y-std, ls='--', c=c2)

plt.title('Mean result from 2nn')
plt.xlabel('disorder strength, $W$')
plt.ylabel('Intrinsic dimension')

plt.legend()

plt.savefig('2nn_mean_12_14_may8.png', dpi=500)
 
