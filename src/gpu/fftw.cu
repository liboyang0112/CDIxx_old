#include "fftw.h"
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
      sz = row*column*sizeof(cufftDoubleComplex);
      cudaFree(cudaData);
      Check(cudaMalloc((void**)&cudaData, sz));
      cufftPlan2d ( plan, row, column, CUFFT_Z2Z);
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

