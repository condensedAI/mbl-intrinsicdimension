from nn2_new import *
import argparse
import os
import numpy as np

def run(L, first_seed, output):
    # Perform 2nn
    if L < 12:
        single = False
    else:
        single = True
    IDs = nn2_new(L, first_seed, single)

    np.save(output + '2nn-%d-%d' % (L, first_seed), IDs, allow_pickle=True)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 2NN on Hubbard chain Eigenvectors")
    parser.add_argument('output', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, default=8, help='The system size (L)')
    parser.add_argument('-S', type=int, help='The seed number')
    args = parser.parse_args()

    # Make sure the output directory exists
    try:
        os.makedirs(args.output, exist_ok = True)
    except:
        print("Directory '%s' could not be created" % args.output)
    
    run(args.L, args.S, args.output)

    