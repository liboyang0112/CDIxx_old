#include <fftw3-mpi.h>
#include "fftw.h"
using namespace cv;
void fftw_init(){
  fftw_init_threads();
  fftw_plan_with_nthreads(3);
}

Mat* fftw ( Mat* in, Mat *out = 0, bool isforward = 1)
{
  fftw_plan plan_forward; 
  int row = in->rows;
  int column = in->cols;
  Real ratio = 1./sqrt(row*column);
  if(out == 0) out = new Mat(row,column,float_cv_format(2));
    
  plan_forward = fftw_plan_dft_2d ( row, column, (fftw_format*)in->data, (fftw_format*)out->data, isforward?FFTW_FORWARD:FFTW_BACKWARD, FFTW_ESTIMATE );
    
  fftw_execute ( plan_forward );

  for(int i = 0; i < out->total() ; i++){
    ((fftw_format*)out->data)[i][0]*=ratio;
    ((fftw_format*)out->data)[i][1]*=ratio;
  } //normalization
  fftw_destroy_plan ( plan_forward );
  return out;
}

