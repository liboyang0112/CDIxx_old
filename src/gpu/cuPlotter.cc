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
  std::string fname = label;
  if(fname.find(".")==std::string::npos) fname+=".png";
  printf("written to file %s\n", fname.c_str());
  if(islog){
    Mat* tmp = (Mat*)cv_cache;
	  Mat dst8 = Mat::zeros(tmp->size(), CV_8U);
	  normalize(*tmp, *tmp, 0, 255, NORM_MINMAX);
	  convertScaleAbs(*tmp, dst8);
	  applyColorMap(dst8, dst8, COLORMAP_TURBO);
	  imwrite(fname,dst8);
  }else
    imwrite(fname, *(Mat*)cv_cache);
}
void cuPlotter::plotFloat(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog){
  cuPlotter::processFloatData(cudaData,m,isFrequency,decay,islog);
  std::string fname = label;
  if(fname.find(".")==std::string::npos) fname+=".png";
  printf("written to file %s\n", fname.c_str());
  if(islog){
    Mat* tmp = (Mat*)cv_cache;
	  Mat dst8 = Mat::zeros(tmp->size(), CV_8U);
	  normalize(*tmp, *tmp, 0, 255, NORM_MINMAX);
	  convertScaleAbs(*tmp, dst8);
	  applyColorMap(dst8, dst8, COLORMAP_TURBO);
	  imwrite(fname,dst8);
  }else
    imwrite(fname, *(Mat*)cv_cache);
}
cuPlotter::~cuPlotter(){
  freeCuda();
  delete (Mat*)cv_cache;
}

