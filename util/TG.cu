#include <complex>
#include <tbb/tbb.h>
#include <fftw3-mpi.h>
# include <cassert>
# include <stdio.h>
# include <time.h>
# include <random>

#include <stdio.h>
#include "fftw.h"
#include <iostream>
#include <fstream>
#include <libconfig.h++>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "cufft.h"

#include "common.h"
#include <ctime>

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

// This code only apply to image with height and width in power of 2, i.e. ... 256, 512, 1024, .... due to Cuda restrictions.
//#define Bits 16
__device__ __constant__ double cuda_beta_HIO;
__device__ __constant__ int cuda_row;
__device__ __constant__ int cuda_column;
__device__ __constant__ int cuda_rcolor;
__device__ __constant__ double cuda_scale;
using namespace cv;
double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

double gaussian_norm(double x, double y, double sigma){
  return 1./(2*pi*sigma*sigma)*gaussian(x,y,sigma);
}

enum mode {MOD2,MOD, REAL, IMAG, PHASE};
/******************************************************************************/
template<typename functor, typename format=fftw_complex>
void imageLoop(Mat* data, void* arg, bool isFrequency = 0){
  int row = data->rows;
  int column = data->cols;
  format *rowp;
  functor* func = static_cast<functor*>(arg);
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = data->ptr<format>(x);
    for(int y = 0; y<column; y++){
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      (*func)(targetx, targety , rowp[y]);
    }
  }
}
template<typename functor, typename format1, typename format2>
void imageLoop(Mat* data, Mat* dataout, void* arg, bool isFrequency = 0){
  int row = data->rows;
  int column = data->cols;
  format1 *rowp;
  format2 *rowo;
  functor* func = static_cast<functor*>(arg);
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = data->ptr<format1>(x);
    rowo = dataout->ptr<format2>(targetx);
    for(int y = 0; y<column; y++){
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      (*func)(targetx, targety , rowp[y], rowo[targety]);
    }
  }
}
/******************************************************************************/
double getVal(mode m, fftw_complex &data){
  complex<double> &tmpc = *(complex<double>*)(data);
  switch(m){
    case MOD:
      return std::abs(tmpc);
      break;
    case MOD2:
      return pow(std::abs(tmpc),2);
      break;
    case IMAG:
      return tmpc.imag();
      break;
    case PHASE:
      if(std::abs(tmpc)==0) return 0;
      return (std::arg(tmpc)+pi)/2/pi;
      break;
    default:
      return tmpc.real();
  }
}
double getVal(mode m, double &data){
  return data;
}
template<typename T=fftw_complex>
Mat* convertFromComplexToInteger(Mat *fftwImage, Mat* opencvImage = 0, mode m = MOD, bool isFrequency = 0, double decay = 1, const char* label= "default",bool islog = 0){
  pixeltype* rowo;
  T* rowp;
  int row = fftwImage->rows;
  int column = fftwImage->cols;
  if(!opencvImage) opencvImage = new Mat(row,column,format_cv);
  int tot = 0;
  double max = 0;
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowo = opencvImage->ptr<pixeltype>(targetx);
    rowp = fftwImage->ptr<T>(x);
    for(int y = 0; y<column; y++){
      double target = getVal(m, rowp[y]);
      if(max < target) max = target;
      if(target<0) target = -target;
      if(islog){
        if(target!=0)
          target = log2(target)*rcolor/log2(rcolor)+rcolor;
	if(target < 0) target = 0;
	
      }
      else target*=rcolor*decay;

      tot += (int)target;
      if(target>=rcolor) {
	      //printf("larger than maximum of %s png %f\n",label, target);
	      target=rcolor-1;
	      //target=0;
      }
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      rowo[targety] = floor(target);
      //if(opencv_reverted) rowp[targety] = rcolor - 1 - rowp[targety];
      //rowp[targety] = rcolor - 1 - rowp[targety];
    }
  }
  printf("total intensity %s: %d, max: %f\n", label, tot, max);
  return opencvImage;
}

__global__ void applyNorm(cufftDoubleComplex* data){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x + y*cuda_row;
  data[index].x*=1./sqrtf(cuda_row*cuda_column);
  data[index].y*=1./sqrtf(cuda_row*cuda_column);
}

int main(int argc, char** argv )
{
    int row = 512;
    int column = 512;
    //These are in mm;
    double lambda = 800e-6;
    double dhole = 2; // distance between two holes of pump light
    double focus = 20;
    double pixelsize = 3e-3;
    double spotSize = 60e-3;
    double dn = 1e-5;
    double dx = 0.1;
    double phi0 = dn*dx/lambda;

    int spotpix = spotSize/pixelsize;
    double k = sin(dhole/2/focus)*2*pi/lambda * pixelsize;
    Mat image (row, column, CV_16UC(1), Scalar::all(0));
    Mat imageInput (row, column, CV_64FC2, Scalar::all(0));
    Mat imageTarget (row, column, CV_64FC2, Scalar::all(0));
    fftw_complex* inputField = (fftw_complex*) imageInput.data;
    for(int x = 0; x < row ; x++){
      for(int y = 0; y<column; y++){
	double Emod =  gaussian(x-0.5*row,y-0.5*column,spotpix);
	/*
	double Emodt = 3*gaussian(((double)x)/row-0.1,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.2,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.3,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.4,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.6,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.7,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.8,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.9,((double)y)/column-0.5,0.01);
		     */
	double phase = cos(k * x)*phi0;
        inputField[x+y*row][0] = Emod*sin(phase);
        inputField[x+y*row][1] = Emod*cos(phase);
      }
    }
    fftw(&imageInput, &imageTarget, 1);
    convertFromComplexToInteger(&imageInput, &image, MOD2,0);
    imwrite("inputIntensity.png",image);
    convertFromComplexToInteger(&imageInput, &image, PHASE,0);
    imwrite("inputPhase.png",image);
    convertFromComplexToInteger(&imageTarget, &image, MOD2,1,1e6);
    imwrite("targetField.png",image);
    return 0;
}
