//
// CUDA implementation of Total Variation Filter
// Implementation of Nonlinear total variation based noise removal algorithms : 10.1016/0167-2789(92)90242-F
//
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cudaConfig.h"
#include "common.h"

#define BLOCK_SIZE      16
#define FILTER_WIDTH    1      //total width=2n+1 
#define FILTER_HEIGHT   1       

#include <cub/device/device_reduce.cuh>
using namespace std;

// Run Total Variation Filter on GPU

Real *d_output, *d_bracket, *d_lambdacore, *lambda;
const short tilewidth=BLOCK_SIZE+2*FILTER_HEIGHT;
size_t  temp_storage_bytes = 0;
void  *d_temp_storage = NULL;
static int rows, cols;
size_t sz;

struct CustomSum
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a+b;
    }
};
CustomSum sum_op;

template <typename T>
__device__ T minmod(T data1, T data2){
  if(data1*data2<=0) return T(0);  
  if(data1<0) return max(data1,data2);
  return min(data1,data2);
}

__global__ void applyConvolution(Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int tilewidth = kernelwidth*2+blockDim.x;
  extern __shared__ float tile[];
  if(threadIdx.x<blockDim.x/2+kernelwidth && threadIdx.y<blockDim.y/2+kernelheight)
    tile[threadIdx.x*(tilewidth)+threadIdx.y]=(x>=kernelwidth && y>=kernelheight)?input[(x-kernelwidth)*cuda_column+y-kernelheight]:0;
  if(threadIdx.x>=blockDim.x/2-kernelwidth && threadIdx.y<blockDim.y/2+kernelheight)
    tile[(threadIdx.x+2*kernelwidth)*(tilewidth)+threadIdx.y]=(x<cuda_row-kernelwidth && y>=kernelheight)?input[(x+kernelwidth)*cuda_column+y-kernelheight]:0;
  if(threadIdx.x<blockDim.x/2+kernelwidth && threadIdx.y>=blockDim.y/2-kernelheight)
    tile[threadIdx.x*(tilewidth)+threadIdx.y+2*kernelheight]=(x>=kernelwidth && y<cuda_column-kernelheight)?input[(x-kernelwidth)*cuda_column+y+kernelheight]:0;
  if(threadIdx.x>=blockDim.x/2-kernelwidth && threadIdx.y>=blockDim.y/2-kernelheight)
    tile[(threadIdx.x+2*kernelwidth)*(tilewidth)+threadIdx.y+2*kernelheight]=(x<cuda_row-kernelwidth && y<cuda_column-kernelheight)?input[(x+kernelwidth)*cuda_column+y+kernelheight]:0;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  __syncthreads();
  int Idx = (threadIdx.x)*(tilewidth) + threadIdx.y;
  int IdxK = 0;
  Real n_output = 0;
  for(int x = -kernelwidth; x <= kernelwidth; x++){
    for(int y = -kernelheight; y <= kernelheight; y++){
      n_output+=tile[Idx++]*kernel[IdxK++];
    }
    Idx+=tilewidth-2*kernelheight-1;
  }
  output[index] = n_output;
}

template <typename T>
__global__ void calcBracketLambda(T *srcImage, T *bracket, T* u0, T* lambdacore, T noiseLevel)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  extern __shared__ float tile[];
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y]=(x>=FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y]=(x<cuda_row-FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x>=FILTER_WIDTH && y<cuda_column-FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x<cuda_row-FILTER_WIDTH && y<cuda_column-FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  __syncthreads();
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  float dt = 1e-7*noiseLevel;
  float sigmafactor = cuda_rcolor*1e-7*cuda_rcolor/(cuda_row*cuda_column*2);
  int centerIdx = (threadIdx.x+FILTER_WIDTH)*(tilewidth) + threadIdx.y+FILTER_HEIGHT;
  float dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  float dpyU = tile[centerIdx+1]-tile[centerIdx];
  float dmxU = tile[centerIdx]-tile[centerIdx-tilewidth];
  float dmyU = tile[centerIdx]-tile[centerIdx-1];
  float sbracket = 0;
  float denom = sqrt(pow(dpxU,2)+pow(dpyU,2));
  if(denom && x<cuda_row-1 && y<cuda_column-1) lambdacore[index] = ((u0[index+cuda_column]*dpxU+u0[index+1]*dpyU-u0[index]*(dpxU+dpyU))/denom-denom)*sigmafactor;
  else lambdacore[index] = 0;
  denom = sqrt(pow(dpxU,2)+pow(minmod(dpyU,dmyU),2));
  if(denom!=0) sbracket += dpxU/denom;
  denom = sqrt(pow(dpyU,2)+pow(minmod(dpxU,dmxU),2));
  if(denom!=0) sbracket += dpyU/denom;
  centerIdx-=1;
  dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  dpyU = tile[centerIdx+1]-tile[centerIdx];
  dmxU = tile[centerIdx]-tile[centerIdx-tilewidth];
  denom = sqrt(pow(dpyU,2)+pow(minmod(dpxU,dmxU),2));
  if(denom != 0) sbracket -= dpyU/denom;
  centerIdx-=tilewidth-1;
  dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  dpyU = tile[centerIdx+1]-tile[centerIdx];
  dmyU = tile[centerIdx]-tile[centerIdx-1];
  denom = sqrt(pow(dpxU,2)+pow(minmod(dpyU,dmyU),2));
  if(denom != 0) sbracket -= dpxU/denom;
  bracket[index] = dt*sbracket;
}

template <typename T>
__global__ void tvFilter(T *srcImage, T *bracket, T* u0, T* slambda)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  srcImage[index]+=bracket[index]-(*slambda)*(srcImage[index]-u0[index]);
}

void inittvFilter(int row, int col){
  rows = row;
  cols = col;
  sz = rows * cols * sizeof(Real);
  // Allocate device memory
  cudaMalloc(&d_output,sz);
  cudaMalloc(&d_bracket,sz);
  cudaMalloc(&d_lambdacore,sz);
  cudaMalloc(&lambda,sizeof(Real));
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_lambdacore, lambda, rows*cols, sum_op, 0));
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
}
Real* tvFilterWrap(Real* d_input, Real noiseLevel, int nIters){
  gpuErrchk(cudaMemcpy(d_output,d_input,sz,cudaMemcpyDeviceToDevice));
	for(int i = 0; i<nIters; i++){
    calcBracketLambda<<<numBlocks,threadsPerBlock,sizeof(float)*(tilewidth)*(tilewidth)>>>(d_output, d_bracket, d_input, d_lambdacore, noiseLevel);
    gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, sz, d_lambdacore, lambda, rows*cols, sum_op, 0));
    tvFilter<<<numBlocks,threadsPerBlock>>>(d_output, d_bracket, d_input, lambda);
	}
  cudaMemcpy(d_input,d_output,sz,cudaMemcpyDeviceToDevice);
	return d_input;
}
void tvFilter(Real* input, Real noiseLevel, int nIters)
{
	Real* d_input;
	gpuErrchk(cudaMalloc(&d_input,sz));
  cudaMemcpy(d_input,input,sz,cudaMemcpyHostToDevice);
	tvFilterWrap(d_input, noiseLevel, nIters);
  cudaMemcpy(input,d_input,sz,cudaMemcpyDeviceToHost);
  cudaFree(d_input);
}
