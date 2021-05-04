#!/bin/bash
savedir="/home/projects/ku_00067/scratch/ising/data/"


for i in `seq 0 999`
do
	qsub -v output=$savedir,L=10,S=$i, W=[1.0, 1.44] -N Ising-L-10-T-$T controlscript.pbs
done