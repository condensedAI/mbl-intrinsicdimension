#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fullresults/"

savedir="fullresults/"


for seed in {0..10..10}
do
	qsub -v output=$savedir,L=$16,S=$seed, -N r-hubb-L-$16-S-$seed R16_controlscript.pbs
done
