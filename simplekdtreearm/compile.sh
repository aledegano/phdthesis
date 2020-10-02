#!/bin/bash

nvcc --compiler-options '-fPIC' -std=c++11 -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcublas -lcublas_device -lcudadevrt -lcudart -lcudart_static -lcufft -lcufftw -lcurand -lcusparse CudaFKDTree.cu powerFKD.cc -o powerFKD
