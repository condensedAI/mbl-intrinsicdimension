#!/bin/bash
savedir="/home/projects/ku_00067/scratch/ising/data/"

for T in `seq 4.5 -0.1 1`
do
	qsub -v output=$savedir,L=10,T=$T,N=10000 -N Ising-L-10-T-$T controlscript.pbs
	qsub -v output=$savedir,L=40,T=$T,N=10000 -N Ising-L-40-T-$T controlscript.pbs
	qsub -v output=$savedir,L=80,T=$T,N=10000 -N Ising-T-80-T-$T controlscript.pbs
done

