#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__
#include <iostream>
#include "format.h"
#include "cudaDefs.h"
#include <curand_kernel.h>
#define FFTformat CUFFT_C2C
#define FFTformatR2C CUFFT_R2C
#define myCufftExec cufftExecC2C
#define myCufftExecR2C cufftExecR2C
__global__ void add(Real* a, Real* b, Real c = 1);
__global__ void applyNorm(complexFormat* data, Real factor);
__global__ void applyNorm(Real* data, Real factor);
__global__ void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling);
__global__ void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col);
__global__ void applyConvolution(Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight);
__global__ void getMod(Real* mod, complexFormat* amp);
__global__ void getMod2(Real* mod, complexFormat* amp);
__global__ void applyPoissonNoise(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
__global__ void applyPoissonNoise_WO(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
__global__ void initRand(curandStateMRG32k3a *state);
__global__ void fillRedundantR2C(complexFormat* data, complexFormat* dataout, Real factor);
__global__ void applyMod(complexFormat* source, Real* target, Real *bs = 0, bool loose=0, int iter = 0, int noiseLevel = 0);
__global__ void add(complexFormat* a, complexFormat* b, Real c = 1);
__global__ void applyRandomPhase(complexFormat* wave, Real* beamstop, curandStateMRG32k3a *state);
void opticalPropagate(complexFormat* field, Real lambda, Real d, Real imagesize, int rows, int cols);
template <typename T>
__global__ void cudaConvertFO(T* data){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= (cuda_row>>1) || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int indexp = (x+(cuda_row>>1))*cuda_column + (y >= (cuda_column>>1)? y-(cuda_column>>1): (y+(cuda_column>>1)));
  T tmp = data[index];
  data[index]=data[indexp];
  data[indexp]=tmp;
}

template <typename T>
__global__ void crop(T* src, T* dest, int row, int col){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= cuda_row || y >= cuda_column) return;
	int index = x*cuda_column + y;
	int targetindex = (x+(row-cuda_row)/2)*col + y+(col-cuda_column)/2;
	dest[index] = src[targetindex];
}

template <typename T>
__global__ void pad(T* src, T* dest, int row, int col){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= cuda_row || y >= cuda_column) return;
	int marginx = (cuda_row-row)/2;
	int marginy = (cuda_column-col)/2;
	int index = x*cuda_column + y;
	if(x < marginx || x >= row+marginx || y < marginy || y >= col+marginy){
		dest[index] = 0;
		return;
	}
	int targetindex = (x-marginx)*col + y-marginy;
	dest[index] = src[targetindex];
}

template <typename T>
__global__ void refine(T* src, T* dest, int refinement){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= cuda_row || y >= cuda_column) return;
	int index = x*cuda_column + y;
	int indexlu = (x/refinement)*(cuda_row/refinement) + y/refinement;
	int indexld = (x/refinement)*(cuda_row/refinement) + y/refinement+1;
	int indexru = (x/refinement+1)*(cuda_row/refinement) + y/refinement;
	int indexrd = (x/refinement+1)*(cuda_row/refinement) + y/refinement+1;
	Real dx = Real(x%refinement)/refinement;
	Real dy = Real(y%refinement)/refinement;
	dest[index] = 
		src[indexlu]*(1-dx)*(1-dy)
		+((y<cuda_column-refinement)?src[indexld]*(1-dx)*dy:0)
		+((x<cuda_row-refinement)?src[indexru]*dx*(1-dy):0)
		+((y<cuda_column-refinement&&x<cuda_row-refinement)?src[indexrd]*dx*dy:0);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class rect{
  public:
    int startx;
    int starty;
    int endx;
    int endy;
    __device__ __host__ bool isInside(int x, int y){
      if(x > startx && x <= endx && y > starty && y <= endy) return true;
      return false;
    }
};
class C_circle{
  public:
    int x0;
    int y0;
    Real r;
    __device__ __host__ bool isInside(int x, int y){
      Real dr = sqrt(pow(x-x0,2)+pow(y-y0,2));
      if(dr < r) return true;
      return false;
    }
};
template <typename sptType>
__global__ void createMask(Real* data, sptType* spt, bool isFrequency=0){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
  }
  data[index]=spt->isInside(x,y);
}
#endif
