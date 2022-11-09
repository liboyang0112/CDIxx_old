#include "cudaDefs.h"
__device__ __constant__ Real cuda_beta_HIO;
__device__ __constant__ int cuda_row;
__device__ __constant__ int cuda_column;
__device__ __constant__ int cuda_rcolor;
__device__ __constant__ Real cuda_scale;
__device__ __constant__ int cuda_totalIntensity;
__device__ __constant__ int cuda_norm;
dim3 numBlocks;
const dim3 threadsPerBlock(16,16);
complexFormat *cudaData = 0;
cufftHandle *plan; 
