#ifndef __CUDADEFS_H__
#define __CUDADEFS_H__
#include "format.h"
#include <cufft.h>
using complexFormat=cufftComplex;
extern const dim3 threadsPerBlock;
extern dim3 numBlocks;
extern __device__ __constant__ Real cuda_beta_HIO;
extern __device__ __constant__ int cuda_row;
extern __device__ __constant__ int cuda_column;
extern __device__ __constant__ int cuda_rcolor;
extern __device__ __constant__ Real cuda_scale;
extern __device__ __constant__ int cuda_totalIntensity;
extern __device__ __constant__ int cuda_norm;
extern complexFormat *cudaData;
extern cufftHandle *plan; 
#endif
