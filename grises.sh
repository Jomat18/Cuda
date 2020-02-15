#!/bin/sh
#PBS -N escala_grises
#PBS -l nodes=n008:ppn=16:gpus=1
#PBS -o grises_test.out
#PBS -e grises_error.err

cd $PBS_O_WORKDIR

nvcc --std=c++11 grayscale.cu -o grayscale
./grayscale


