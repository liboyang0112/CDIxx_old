#include "cuPlotter.h"
#include "opencv2/opencv.hpp"
using namespace cv;
void cuPlotter::init(int rows_, int cols_){
  rows=rows_;
  cols=cols_;
  Mat *tmp = new Mat(rows_, cols_, CV_16UC1, Scalar(0));
  cv_cache = tmp;
  cv_data = tmp->data;
  initcuData(rows*cols*sizeof(pixeltype));
}
void cuPlotter::plotComplex(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog){
  cuPlotter::processComplexData(cudaData,m,isFrequency,decay,islog);
  imwrite(std::string(label)+".png", *(Mat*)cv_cache);
}
void cuPlotter::plotFloat(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog){
  cuPlotter::processFloatData(cudaData,m,isFrequency,decay,islog);
  imwrite(std::string(label)+".png", *(Mat*)cv_cache);
}
cuPlotter::~cuPlotter(){
  freeCuda();
  delete (Mat*)cv_cache;
}

