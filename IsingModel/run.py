import argparse
import os
import numpy as np
from WolffAlgorithm import WolffMonteCarlo

def run(L, T, output, numSamplesPerT):
    # Initialize a new simulator
    sim = WolffMonteCarlo(L=L, T=5, method="DFS")

    # Loop over a fixed set of temperatures
    print("Generating samples for L = %d at T = %.5f"%(L,T))

    # Set temperature
    sim.set_T(T)

    # For storing all of the configurations
    res = []
    for s in range(numSamplesPerT):

        # Keep flipping sites, until we flipped at least L^2 of them
        c = 0
        while c < 1:
        	to_flip = sim.step()
        	c = c + len(to_flip) / L / L

        # The first half of the flips are to equilibrate, the rest are
        # good samples
        if s >= numSamplesPerT//2:
        	res.append(np.concatenate([[T], -1 + 2 * sim.state.reshape(-1)]))

    np.save(output + '/%d-%.5f' % (L, T), res, allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run (Wolff) Monte Carlo to generate Ising model snapshots")
    parser.add_argument('output', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, default=10, help='The system size (LxL)')
    parser.add_argument('-N', type=int, default=20000, help='The number of samples per temperature')
    parser.add_argument('-T', type=float, help='The temperature at which to generate samples')
    args = parser.parse_args()

    # Make sure the output directory exists
    try:
        os.makedirs(args.output, exist_ok = True)
    except:
        print("Directory '%s' could not be created" % args.output)

    run(args.L, args.T, args.output, args.N)
