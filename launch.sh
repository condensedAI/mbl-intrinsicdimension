#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fullresults/"

savedir="fullresults/"

L=14

for i in {0..10000..10}
do
	qsub -v output=$savedir,L=$L,S=$i, -N hubbard-L-$L-S-$i controlscript.pbs
done
