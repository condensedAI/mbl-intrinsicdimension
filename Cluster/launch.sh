#!/bin/bash
savedir="../results_NEW/"


for i in `seq 0 2`
do
	qsub -v output=$savedir,L=12,S=$i, -N hubbard-L-$L-S-$S controlscript.pbs
done