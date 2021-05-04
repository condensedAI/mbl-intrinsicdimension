from scipy.spatial import distance_matrix
import numpy as np
import time

data = np.random.rand(924*924).reshape(924,924)



starttime = time.time()
dist_matrix = np.zeros((924, 924))
for i, eigvec1 in enumerate(data):
	for j, eigvec2 in enumerate(data):
		if j <= i:
			pass
		else:
			distance = sum(abs((eigvec1-eigvec2)))
			dist_matrix[i,j], dist_matrix[j,i] = distance, distance

print("This part took %.5f seconds"%(time.time() - starttime))


starttime = time.time()

a = distance_matrix(data,data, 1)

print("This second part took %.5f seconds"%(time.time() - starttime))



print(np.allclose(dist_matrix, a))