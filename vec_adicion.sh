#!/bin/sh
#PBS -N adicion_vectores
#PBS -l nodes=n008:ppn=16:gpus=1
#PBS -o vec_test.out    
#PBS -e vec_error.err

cd $PBS_O_WORKDIR

nvcc vecAdd.cu -o vecAdd
cuda-memcheck ./vecAdd
