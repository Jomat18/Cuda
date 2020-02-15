#!/bin/sh
#PBS -N blur_imagen
#PBS -l nodes=n008:ppn=16:gpus=1
#PBS -o blur_test.out
#PBS -e blur_error.err

cd $PBS_O_WORKDIR

nvcc --std=c++11 blur_imagen.cu -o blur
./blur

