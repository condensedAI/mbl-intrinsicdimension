#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fullresults/"

savedir="fullresults/"


for L in {14..14..2}
do
	qsub -v output=$savedir,L=$L,S=10000, -N hubbard-L-$L-S-$i R_controlscript.pbs
done
