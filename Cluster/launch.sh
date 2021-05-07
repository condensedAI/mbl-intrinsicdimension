#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/results_NEW/"

L=14

for i in `seq 0 999`
do
	qsub -v output=$savedir,L=$L,S=$i, -N hubbard-L-$L-S-$i controlscript.pbs
done
