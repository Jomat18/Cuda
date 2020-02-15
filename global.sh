#!/bin/sh
#PBS -N mul_global
#PBS -l nodes=n008:ppn=16:gpus=1
#PBS -o global.out
#PBS -e global.err

cd $PBS_O_WORKDIR

nvcc mul_global.cu -o global
./global

