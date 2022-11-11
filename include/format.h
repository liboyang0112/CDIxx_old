#ifndef __FORMAT_H__
#define __FORMAT_H__
#define Bits 16
#include <complex>

#if Bits==12
using pixeltype=uint16_t;
#elif Bits==16
using pixeltype=uint16_t;
#else
using pixeltype=uchar;
#endif
using Real=float;
using fftw_format=std::complex<Real>;
int float_cv_format(int i);
#endif
