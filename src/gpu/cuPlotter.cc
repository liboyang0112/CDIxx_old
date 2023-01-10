#include "cuPlotter.h"
#include "opencv2/opencv.hpp"
#include "opencv2/phase_unwrapping/histogramphaseunwrapping.hpp"
using namespace cv;
void cuPlotter::init(int rows_, int cols_){
  rows=rows_;
  cols=cols_;
  Mat *tmp = new Mat(rows_, cols_, CV_16UC1, Scalar(0));
  cv_cache = tmp;
  cv_data = tmp->data;
  Mat *tmpfloat = new Mat(rows_, cols_, CV_32FC1, Scalar(0));
  cv_float_cache = tmpfloat;
  cv_float_data = tmpfloat->data;
  Mat *tmpcomplex = new Mat(rows_, cols_, CV_32FC2, Scalar(0));
  cv_complex_cache = tmpcomplex;
  cv_complex_data = tmpcomplex->data;
  initcuData(rows*cols);
}
void cuPlotter::plotComplex(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog){
  cuPlotter::processComplexData(cudaData,m,isFrequency,decay,islog);
  plot(label, islog);
}
void cuPlotter::plotFloat(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog){
  cuPlotter::processFloatData(cudaData,m,isFrequency,decay,islog);
  plot(label, islog);
}
void cuPlotter::saveComplex(void* cudaData, const char* label){
  saveComplexData(cudaData);
  FileStorage fs(label,FileStorage::WRITE);
  fs<<"data"<<*((Mat*)cv_float_cache);
  fs.release();
}
void cuPlotter::saveFloat(void* cudaData, const char* label){
  saveFloatData(cudaData);
  imwrite(std::string(label)+".tiff",*((Mat*)cv_float_cache));
      /*
  FileStorage fs(label,FileStorage::WRITE);
  fs<<"data"<<*((Mat*)cv_float_cache);
  fs.release();
  */
}
void cuPlotter::plotPhase(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog){
  cuPlotter::processPhaseData(cudaData,m,isFrequency,decay);
  cv::phase_unwrapping::HistogramPhaseUnwrapping::Params pars;
  pars.height = cols;
  pars.width = rows;
  auto uwrap = phase_unwrapping::HistogramPhaseUnwrapping::create(pars);
  uwrap->unwrapPhaseMap(*(Mat*)cv_float_cache, *(Mat*)cv_float_cache);
  for(int i = 0; i < rows*cols; i++)
    ((pixeltype*)cv_data)[i] = std::min(int((((Real*)cv_float_data)[i]+Real(M_PI))*rcolor/phaseMax),rcolor-1);
  plot(label, islog);
}
void cuPlotter::plot(const char* label, bool iscolor){
  std::string fname = label;
  if(fname.find(".")==std::string::npos) fname+=".png";
  printf("written to file %s\n", fname.c_str());
  if(iscolor){
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

