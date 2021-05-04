#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/results_NEW/"


for i in `seq 0 20`
do
	qsub -v output=$savedir,L=10,S=$i, -N hubbard-L-10-S-$S controlscript.pbs
done