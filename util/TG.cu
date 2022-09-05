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

/******************************************************************************/

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
