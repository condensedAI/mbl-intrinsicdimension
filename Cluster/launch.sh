#!/bin/bash
savedir="../results_NEW/"


for i in `seq 0 20`
do
	qsub -v output=$savedir,L=$L,S=$i, -N hubbard-L-$L-S-$S controlscript.pbs
done