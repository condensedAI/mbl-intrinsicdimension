#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fullresults/"

savedir="fullresults/"

L=14

for first_seed in {0..100..100}
do
	qsub -v output=$savedir,L=$L,S=$100,K=$first_seed, -N r-hubb-L-$L-S-$first_seed R_controlscript.pbs
	done
