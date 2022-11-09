#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__
#include <iostream>
#include <cufft.h>
#include "format.h"
#include "cudaDefs.h"
static const decltype(CUFFT_Z2Z) FFTformat=CUFFT_C2C;
template<typename... Args>
auto myCufftExec(Args... arg){
	return cufftExecC2C(arg...);
}
void init_cuda_image(int rows, int cols, int rcolor=65536, Real scale=1);
__global__ void applyNorm(complexFormat* data, double factor);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif
