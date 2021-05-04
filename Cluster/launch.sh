#!/bin/bash
savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/results_NEW/"


for i in `seq 0 999`
do
	qsub -v output=$savedir,L=10,S=$i, W=[1.0, -N hubbard-L-10-S-$S controlscript.pbs
done