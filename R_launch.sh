#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fullresults/"

savedir="fullresults/"

L=16
S=1
for first_seed in {0..10000..1}
do
	qsub -v output=$savedir,L=$L,S=$S,K=$first_seed, -N r-hubb-L-$L-S-$first_seed R_controlscript.pbs
	done
