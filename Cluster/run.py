from nn2_cluster import *
import argparse
import os
import numpy as np

def run(L, seed, output):
    # List filenames
    filenames = listFilenames(L=12, seed=seed)

    # Get eigencevtors
    eigenvectors = load_many_eigs(filenames)

    # Perform 2nn
    IDs = [nn2(eigs) for eigs in eigenvectors]

    np.save(output + '/%d-%d' % (L, seed), IDs, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 2NN on Hubbard chain Eigenvectors")
    parser.add_argument('output', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, default=12, help='The system size (L)')
    parser.add_argument('-S', type=int, help='The seed number')
    args = parser.parse_args()

    # Make sure the output directory exists
    try:
        os.makedirs(args.output, exist_ok = True)
    except:
        print("Directory '%s' could not be created" % args.output)
    
    run(args.L, args.S, args.output)

    
