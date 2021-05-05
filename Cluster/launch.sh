#!/bin/bash
savedir="../results_NEW/"


for i in `seq 0 20`
do
	qsub -v output=$savedir,L=12,S=$i, -N hubbard-L-10-S-$S controlscript.pbs
done