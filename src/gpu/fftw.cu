#include "format.h"
#include "cudaConfig.h"
#include <cufftw.h>
#include <iostream>
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

void fftw_init(){
  
}
static size_t sz;

__global__ void applyNorm(complexFormat* data, double factor){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  data[index].x*=factor;
  data[index].y*=factor;
}

Mat* fftw ( Mat* in, Mat *out, bool isforward, Real ratio)
{
  int row = in->rows;
  int column = in->cols;
  if(ratio==0) ratio = 1./sqrt(row*column);
  if(out == 0) out = new Mat(row,column,float_cv_format(2));

  if(cudaData==0) {
    sz = row*column*sizeof(complexFormat);
    gpuErrchk(cudaMalloc((void**)&cudaData, sz));
    plan = new cufftHandle();
    cufftPlan2d ( plan, row, column, FFTformat);
  }else{
    if(sz!=row*column*sizeof(complexFormat)){
      printf("reconfiguring CUFFT\n");
      sz = row*column*sizeof(complexFormat);
      cudaFree(cudaData);
      gpuErrchk(cudaMalloc((void**)&cudaData, sz));
      cufftPlan2d ( plan, row, column, FFTformat);
    }
  }
  gpuErrchk(cudaMemcpy(cudaData, in->data, sz, cudaMemcpyHostToDevice));
    
  myCufftExec( *plan, cudaData,cudaData, isforward? CUFFT_FORWARD: CUFFT_INVERSE);

  applyNorm<<<threadsPerBlock,numBlocks>>>(cudaData, ratio);

  gpuErrchk(cudaMemcpy(out->data, cudaData, sz, cudaMemcpyDeviceToHost));
  
  return out;
}

