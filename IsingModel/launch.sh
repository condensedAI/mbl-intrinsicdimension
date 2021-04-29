#!/bin/bash
savedir="/home/projects/ku_00067/scratch/ising/data/"

for T in `seq 4.5 -0.1 1`
do
	echo "qsub -v output=$savedir,L=40,N=10000 -N Ising-T-$T controlscript.pbs"
done

