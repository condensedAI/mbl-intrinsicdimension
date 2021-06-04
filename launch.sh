#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fullresults/"

L=14

for i in {0..1000..100}
do
	qsub -v output=$savedir,L=$L,S=$i, -N hubbard-L-$L-S-$i controlscript.pbs
done
