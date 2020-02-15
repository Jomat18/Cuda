#!/bin/sh
#PBS -N propiedades
#PBS -l nodes=n008:ppn=16:gpus=1
#PBS -o prop_test.out
#PBS -e prop_error.err

cd $PBS_O_WORKDIR

nvcc propiedades.cu -o propiedades
./propiedades


