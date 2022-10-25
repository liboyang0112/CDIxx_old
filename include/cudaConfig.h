#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__
#include <cufft.h>
#include "format.h"
const dim3 threadsPerBlock(16,16);
static const decltype(CUFFT_Z2Z) FFTformat=CUFFT_C2C;
template<typename... Args>
auto myCufftExec(Args... arg){
	return cufftExecC2C(arg...);
}
using complexFormat=cufftComplex;
extern __device__ __constant__ Real cuda_beta_HIO;
extern __device__ __constant__ int cuda_row;
extern __device__ __constant__ int cuda_column;
extern __device__ __constant__ int cuda_rcolor;
extern __device__ __constant__ Real cuda_scale;
extern __device__ __constant__ int cuda_totalIntensity;
#endif
