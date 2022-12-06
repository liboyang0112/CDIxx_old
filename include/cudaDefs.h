#ifndef __CUDADEFS_H__
#define __CUDADEFS_H__
#include "format.h"
#include <cufft.h>
#include "memManager.h"
#define cudaF(a) a<<<numBlocks,threadsPerBlock>>>
using complexFormat=cufftComplex;
extern const dim3 threadsPerBlock;
extern dim3 numBlocks;
extern __device__ __constant__ Real cuda_beta_HIO;
extern __device__ __constant__ int cuda_row;
extern __device__ __constant__ int cuda_column;
extern __device__ __constant__ int cuda_rcolor;
extern __device__ __constant__ Real cuda_scale;
extern __device__ __constant__ int cuda_totalIntensity;
extern __device__ __constant__ Real cuda_threshold;
extern complexFormat *cudaData;
extern cufftHandle *plan, *planR2C;

class cuMemManager : public memManager{
  void c_malloc(void*& ptr, size_t sz);
  public:
    cuMemManager():memManager(){};
};
extern cuMemManager memMngr;

#endif
