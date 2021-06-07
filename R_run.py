from R_new import *
import argparse
import os
import numpy as np

def run(L, seeds, first_seed):
	if L < 11:
		single = False
		R(L,seeds,single)
	else:
		single = True
		R(L,seeds,single, first_seed)
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Run 2NN on Hubbard chain Eigenvectors")
	parser.add_argument('output', type=str, help='Output folder of the files')
	parser.add_argument('-L', type=int, default=8, help='The system size (L)')
	parser.add_argument('-S', type=int, help='The number of seeds')
	parser.add_argument('-K', type=int, help='The first seed number')
	args = parser.parse_args()

	# Make sure the output directory exists
	try:
		os.makedirs(args.output, exist_ok = True)
	except:
		print("Directory '%s' could not be created" % args.output)
    
	run(args.L, args.S, args.K)

    
