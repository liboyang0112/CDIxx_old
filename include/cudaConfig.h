#include <cufft.h>
const dim3 threadsPerBlock(16,16);
__device__ __constant__ double cuda_beta_HIO;
__device__ __constant__ int cuda_row;
__device__ __constant__ int cuda_column;
__device__ __constant__ int cuda_rcolor;
__device__ __constant__ double cuda_scale;
__device__ __constant__ int cuda_totalIntensity;
