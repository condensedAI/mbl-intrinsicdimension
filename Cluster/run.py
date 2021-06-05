from make_eigs import constructHamiltonian
import numpy as np
import numpy.linalg
import argparse
import os
import sys

def run(L, startseed, endseed, output):
	for s in range(startseed, endseed):
		data = {}
		for w in np.linspace(0.1,6.1,31):
			H = constructHamiltonian(L = L, W = w, seed=s)
			eigvals, eigvecs = np.linalg.eigh(H)
			data[round(w,2)] = {'eigvals':eigvals, 'eigvecs':eigvecs}
		np.save('{}/eigvecs-and-eigvals-L-{}-seed-{}.npy'.format(output, L, s), data)

if __name__ == '__main__':
	output = sys.argv[1]
	L = int(sys.argv[2])
	Ss = int(sys.argv[3])
	Se = int(sys.argv[4])

	os.makedirs(output, exist_ok = True)
	run(L, Ss, Se, output)
