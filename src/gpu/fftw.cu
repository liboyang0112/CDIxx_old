#include <cufftw.h>
#include "fftw.h"
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

static cufftDoubleComplex *cudaData = 0;
static cufftHandle *plan; 
static size_t sz;

Mat* fftw ( Mat* in, Mat *out = 0, bool isforward = 1)
{
  int row = in->rows;
  int column = in->cols;
  double ratio = 1./sqrt(row*column);
  if(out == 0) out = new Mat(row,column,CV_64FC2);

  if(cudaData==0) {
    sz = row*column*sizeof(cufftDoubleComplex);
    Check(cudaMalloc((void**)&cudaData, sz));
    plan = new cufftHandle();
    cufftPlan2d ( plan, row, column, CUFFT_Z2Z);
  }else{
    if(sz!=row*column*sizeof(cufftDoubleComplex)){
      printf("ERROR: currently cufft only supports single image size to avoid construct and destroy the plan, please check if you are trying to FFT images with different dimensions:\n %lu/(%d*%d*%lu)=%f\n",sz,row,column,sizeof(cufftDoubleComplex),((double)sz)/row/column/sizeof(cufftDoubleComplex));
      printf("FILE: %s, LINE: %d\n",__FILE__, __LINE__);
      exit(0);
    }
  }
  Check(cudaMemcpy(cudaData, in->data, sz, cudaMemcpyHostToDevice));
    
  cufftExecZ2Z( *plan, cudaData,cudaData, isforward? CUFFT_FORWARD: CUFFT_INVERSE);

  Check(cudaMemcpy(out->data, cudaData, sz, cudaMemcpyDeviceToHost));

  for(int i = 0; i < out->total() ; i++){
    ((cufftDoubleComplex*)out->data)[i].x*=ratio;
    ((cufftDoubleComplex*)out->data)[i].y*=ratio;
  } //normalization
  //cufftDestroy(*plan);
  return out;
}

