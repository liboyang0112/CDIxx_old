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

Real getVal(mode m, fftw_format &data);
Real getVal(mode m, Real &data);
template<typename T=fftw_format>
Mat* convertFromComplexToInteger(Mat *fftwImage, Mat* opencvImage = 0, mode m = MOD, bool isFrequency = 0, Real decay = 1, const char* label= "default",bool islog = 0){
  pixeltype* rowo;
  T* rowp;
  int row = fftwImage->rows;
  int column = fftwImage->cols;
  if(!opencvImage) opencvImage = new Mat(row,column,format_cv);
  Real tot = 0;
  Real max = 0;
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowo = opencvImage->ptr<pixeltype>(targetx);
    rowp = fftwImage->ptr<T>(x);
    for(int y = 0; y<column; y++){
      Real target = getVal(m, rowp[y]);
      tot += target;
      if(max < target) max = target;
      if(islog){
        if(target!=0)
          target = log2(target)*rcolor/log2(rcolor)+rcolor;
	if(target < 0) target = 0;
	
      }
      else target*=rcolor*decay;
      if(target<0) target = -target;

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
  printf("total intensity %s: %4.2e, max: %f\n", label, tot/row/column, max);
  return opencvImage;
}

Mat* convertFromIntegerToComplex(Mat &image, Mat* cache = 0, bool isFrequency = 0, const char* label= "default");
Mat* convertFromIntegerToComplex(Mat &image,Mat &phase,Mat* cache = 0);
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
Mat* extend( Mat &src , Real ratio, Real val = 0);
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
