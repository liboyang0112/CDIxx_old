#ifndef __CUPLOTTER_H__
#define __CUPLOTTER_H__

#include "format.h"
enum mode {MOD2,MOD, REAL, IMAG, PHASE, PHASERAD};

class cuPlotter
{
  int rows;
  int cols;
  Real phaseMax = 2*M_PI;
  void *cv_cache;
  void *cv_data; //cv_data = cv_cache->data
  pixeltype *cuCache_data = 0; //cv format
                               //
  void *cv_float_cache;
  void *cv_float_data; //cv_data = cv_cache->data
  Real *cuCache_float_data = 0; //cv format
	public:
  cuPlotter():cuCache_data(0),cv_data(0),cv_cache(0){};
  void init(int rows_, int cols_);
  void initcuData(size_t sz);
  void freeCuda();
  void plotComplex(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0);  //call processData
  void plotFloat(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0);
  void plotPhase(void* cudaData, const mode m=PHASERAD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0); //phase unwrapping
  void processFloatData(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0); //calculate using cuCache_data and copy data to cv_data
  void processComplexData(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0); //calculate using cuCache_data and copy data to cv_data
  void processPhaseData(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1);
  void plot(const char* label, bool islog = 0);
  ~cuPlotter();
};
#endif
