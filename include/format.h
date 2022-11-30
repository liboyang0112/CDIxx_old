#ifndef __FORMAT_H__
#define __FORMAT_H__
#define Bits 16
#include <complex>

#define float_cv_format CV_32FC

#if Bits==12
using pixeltype=uint16_t;
#elif Bits==16
using pixeltype=uint16_t;
#else
using pixeltype=uchar;
#endif
using Real=float;
using fftw_format=std::complex<Real>;
#endif
