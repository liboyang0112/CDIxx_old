#ifndef __COMMON_H__
#define __COMMON_H__
#define Bits 16
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <complex>
#include "fftw3.h"
#include "format.h"

using namespace cv;
// Declare the variables
using namespace std;
static const int mergeDepth = 1; //use it only when input image is integers
#if Bits==12
using pixeltype=uint16_t;
static const int nbits = 12;
static const auto format_cv = CV_16UC1;
#elif Bits==16
using pixeltype=uint16_t;
static const int nbits = 16;
static const auto format_cv = CV_16UC1;
#else
using pixeltype=uchar;
static const int nbits = 8;
static const auto format_cv = CV_8UC1;
#endif
using fftw_format=complex<Real>;
//using inputtype=uchar;
//static const int inputbits = 8;

static const int rcolor = pow(2,nbits);
static bool opencv_reverted = 0;
static const Real scale = 1;

const Real pi = 3.1415927;
enum mode {MOD2,MOD, REAL, IMAG, PHASE};

template<typename functor, typename format=fftw_format>
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
Mat* multiWLGen(Mat* original, Mat* merged, Real m, Real step = 1, Real dphaselambda = 0, Real *spectrum = 0);
Mat* multiWLGenAVG(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT_AC(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT_FFT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT_AC_FFT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_AC_FFT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
template<typename T = complex<Real>>
Mat* convertFO(Mat* mat, Mat* cache = 0){
	int rows = mat->rows;
	int cols = mat->cols;
	if(cache == 0) {
		cache = new Mat();
		mat->copyTo(*cache);
	}
	T *rowi, *rowo;
	for(int x = 0 ; x < rows; x++){
		rowi = mat->ptr<T>(x);
		rowo = cache->ptr<T>((x >= rows/2)? x-rows/2:(x+rows/2));
		for(int y = 0 ; y < cols ; y++){
			rowo[(y >= cols/2)?y-cols/2:(y+cols/2)] = rowi[y];
		}
	}
	return cache;
}

void plotColor(const char* name, Mat* logged);
#endif
