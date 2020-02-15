#!/bin/sh
#PBS -N mul_shared
#PBS -l nodes=n008:ppn=16:gpus=1
#PBS -o compartida.out
#PBS -e compartida.err

cd $PBS_O_WORKDIR

nvcc mul_compartida.cu -o compartida
./compartida


