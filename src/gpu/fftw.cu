#include "format.h"
#include "cudaConfig.h"
#include <cufftw.h>
#include <iostream>
using namespace cv;
using namespace std;

void fftw_init(){
  
}
void Check(cudaError_t status)
{
        if (status != cudaSuccess)
        {
                cout << "行号:" << __LINE__ << endl;
                cout << "错误:" << cudaGetErrorString(status) << endl;
        }
}

static complexFormat *cudaData = 0;
static cufftHandle *plan; 
static size_t sz;

Mat* fftw ( Mat* in, Mat *out, bool isforward, Real ratio)
{
  int row = in->rows;
  int column = in->cols;
  if(ratio==0) ratio = 1./sqrt(row*column);
  if(out == 0) out = new Mat(row,column,float_cv_format(2));

  if(cudaData==0) {
    sz = row*column*sizeof(complexFormat);
    Check(cudaMalloc((void**)&cudaData, sz));
    plan = new cufftHandle();
    cufftPlan2d ( plan, row, column, FFTformat);
  }else{
    if(sz!=row*column*sizeof(complexFormat)){
      printf("reconfiguring CUFFT\n");
      sz = row*column*sizeof(complexFormat);
      cudaFree(cudaData);
      Check(cudaMalloc((void**)&cudaData, sz));
      cufftPlan2d ( plan, row, column, FFTformat);
    }
  }
  Check(cudaMemcpy(cudaData, in->data, sz, cudaMemcpyHostToDevice));
    
  myCufftExec( *plan, cudaData,cudaData, isforward? CUFFT_FORWARD: CUFFT_INVERSE);

  Check(cudaMemcpy(out->data, cudaData, sz, cudaMemcpyDeviceToHost));

  for(int i = 0; i < out->total() ; i++){
    ((complexFormat*)out->data)[i].x*=ratio;
    ((complexFormat*)out->data)[i].y*=ratio;
  } //normalization
  //cufftDestroy(*plan);
  return out;
}

