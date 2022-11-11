#ifndef __COMMON_H__
#define __COMMON_H__
#include <complex>
#include "fftw3.h"
#include "format.h"

// Declare the variables
using namespace std;
static const int mergeDepth = 1; //use it only when input image is integers
//using inputtype=uchar;
//static const int inputbits = 8;

static const int rcolor = pow(2,Bits);
static const Real scale = 1;

const Real pi = 3.1415927;
void init_cuda_image(int rows, int cols, int rcolor=65536, Real scale=1);
void *readComplexImage(const char* name);
void writeComplexImage(const char* name, void* data, int row, int column);
Real *readImage(const char* name, int &row, int &col, bool isFrequency = 0);


#endif
