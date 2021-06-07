#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fullresults/"

savedir="fullresults/"

L=16

for i in {0..0..1}
do
	qsub -v output=$savedir,L=$L,S=$i, -N 2nn-hubb-L-$L-S-$i controlscript.pbs
done
