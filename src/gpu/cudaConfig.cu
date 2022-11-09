#include "cudaConfig.h"
#include "initCuda.h"
#include <iostream>
using namespace std;
void init_cuda_image(int rows, int cols, int rcolor, Real scale){
    cudaMemcpyToSymbol(cuda_row,&rows,sizeof(rows));
    cudaMemcpyToSymbol(cuda_column,&cols,sizeof(cols));
    Real ratio = 1./sqrt(rows*cols);
    cudaMemcpyToSymbol(cuda_norm,&ratio,sizeof(ratio));
    cudaMemcpyToSymbol(cuda_rcolor,&rcolor,sizeof(rcolor));
    cudaMemcpyToSymbol(cuda_scale,&scale,sizeof(scale));
    size_t data_size = rows*cols*sizeof(complexFormat);
    numBlocks.x=(rows-1)/threadsPerBlock.x+1;
    numBlocks.y=(cols-1)/threadsPerBlock.y+1;
    if(!plan){
      plan = new cufftHandle();
    }else{
      cufftDestroy(*plan);
      cudaFree(cudaData);
    }
    cufftPlan2d ( plan, rows, cols, FFTformat);
    gpuErrchk(cudaMalloc((void**)&cudaData, data_size));
};

__global__ void applyNorm(complexFormat* data, Real factor){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  data[index].x*=factor;
  data[index].y*=factor;
}

