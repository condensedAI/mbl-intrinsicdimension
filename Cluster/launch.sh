#!/bin/bash

#for L in 8 
#do
#	savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-8"
#	for startseed in `seq 0 100 9999`
#	do
#		endseed=$(($startseed+99))
#		qsub -v output=$savedir,L=$L,startseed=$startseed,endseed=$endseed -N hubbard-L-$L-S-$startseed controlscript.pbs
#	done
#done
#
#for L in 10
#do
#	savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-10"
#	for startseed in `seq 0 100 9999`
#	do
#		endseed=$(($startseed+99))
#		qsub -v output=$savedir,L=$L,startseed=$startseed,endseed=$endseed -N hubbard-L-$L-S-$startseed controlscript.pbs
#	done
#done

for L in 12
do
	savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-12"
	for startseed in `seq 0 100 9999`
	do
		endseed=$(($startseed+99))
		qsub -v output=$savedir,L=$L,startseed=$startseed,endseed=$endseed -N hubbard-L-$L-S-$startseed controlscript.pbs
	done
done

for L in 14
do
	savedir="/home/projects/ku_00067/scratch/mbl-intrinsicdimension/fulldataset/L-14"
	for startseed in `seq 0 10 9999`
	do
		endseed=$(($startseed+9))
		qsub -v output=$savedir,L=$L,startseed=$startseed,endseed=$endseed -N hubbard-L-$L-S-$startseed controlscript.pbs
	done
done
