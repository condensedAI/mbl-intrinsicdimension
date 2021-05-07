from numpy import load, array, zeros, log, vstack
from numpy.linalg import lstsq
from math import factorial
from scipy.spatial import distance_matrix
import time

def load_eigs_npz(filename='data/results-L-12-W-1.0-seed-42.npz'):
	data = load(filename)
	eigvecs = data[data.files[1]]
	return eigvecs.T

def listFilenames(L = 12,
				  ws = [1.0, 1.55, 2.09, 2.64, 3.18, 3.73, 4.27, 4.82, 5.36, 5.91, 6.45, 7.0],
				   seed = 0):
	files = ['/home/projects/ku_00067/scratch/mbl-intrinsicdimension/data/results-L-{}-W-{}-seed-{}.npz'.format(L,round(w,2),seed) for w in ws]
	
	return files

def load_many_eigs(filenames):
	return array([load_eigs_npz(filename) for filename in filenames], dtype='object')


def nn2(data):
	'''
	Find intrinsic dimension (ID) via 2-nearest-neighbours

	https://www.nature.com/articles/s41598-017-11873-y
	https://arxiv.org/pdf/2006.12953.pdf
	_______________
	Parameters:	eigvecs ---	Returns: Intrinsic Dimension
	
	'''
	N = len(data)
	dist_M = distance_matrix(data,data, p=1)
	
	# table of distances - state and \mu= r_2/r_1
	mu = zeros((N,2))
	for index, line in enumerate(dist_M):

		r1, r2 = sorted(line)[1:3]
		mu[index,0] = index+1
		mu[index,1] = r2/r1
		

	#permutation function
	sigma_i = dict(zip(range(1,len(mu)+1), array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))
	mu = dict(mu)
	#cdf F(mu_{sigma(i)})
	F_i = {}
	for i in mu:
		F_i[sigma_i[i]] = i/N

	#fitting coordinates
	x = log([mu[i] for i in sorted(mu.keys())])
	y = array([1-F_i[i] for i in sorted(mu.keys())])

	#avoid having log(0)
	x = x[y>0]
	y = y[y>0]

	y = -1*log(y)

	#fit line through origin to get the dimension
	d = lstsq(vstack([x, zeros(len(x))]).T, y, rcond=None)[0][0]

	return d