#ifndef __CUPLOTTER_H__
#define __CUPLOTTER_H__

#include "format.h"
enum mode {MOD2,MOD, REAL, IMAG, PHASE};

class cuPlotter
{
  void *cv_cache;
  int rows;
  int cols;
  void *cv_data; //cv_data = cv_cache->data
  pixeltype *cuCache_data = 0; //cv format
	public:
  cuPlotter():cuCache_data(0),cv_data(0),cv_cache(0){};
  void init(int rows_, int cols_);
  void initcuData(size_t sz);
  void freeCuda();
  void plotComplex(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0);  //call processData
  void plotFloat(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0);
  void processFloatData(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0); //calculate using cuCache_data and copy data to cv_data
  void processComplexData(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0); //calculate using cuCache_data and copy data to cv_data
  ~cuPlotter();
};
#endif
