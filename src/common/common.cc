#include "imageReader.h"
#include "readCXI.h"
#include "common.h"
using namespace std;
Real* readImage(const char* name, int &row, int &col, bool isFrequency){
  printf("reading file: %s\n", name);
  Real *data;
  if(string(name).find(".cxi")!=string::npos){
    printf("Input is recognized as cxi file\n");
    Mat *mask;
    Mat imagein = readCXI(name, &mask);  //32FC2, max 65535
    row = imagein.rows;
    col = imagein.cols;
    data = (Real*)ccmemMngr.borrowCache(imagein.total()*sizeof(Real));
    for(int i = 0 ; i < imagein.total() ; i++) data[i] = ((complex<float>*)imagein.data)[i].real()/rcolor;
    return data;
  }
  Mat imagein = imread( name, IMREAD_UNCHANGED  );
  row = imagein.rows;
  col = imagein.cols;
  data = (Real*)ccmemMngr.borrowCache(imagein.total()*sizeof(Real));
  if(imagein.depth() == CV_8U){
    printf("input image nbits: 8, channels=%d\n",imagein.channels());
    if(imagein.channels()>=3){
      Mat image(imagein.rows, imagein.cols, CV_8UC1);
      cv::cvtColor(imagein, image, cv::COLOR_BGR2GRAY);
      for(int i = 0 ; i < image.total() ; i++) data[i] = ((float)(image.data[i]))/255;
    }else{
      for(int i = 0 ; i < imagein.total() ; i++) data[i] = ((float)(imagein.data[i]))/255;
    }
  }else if(imagein.depth() == CV_16U){
    printf("input image nbits: 16\n");
    for(int i = 0 ; i < imagein.total() ; i++) data[i] = ((float)(((uint16_t*)imagein.data)[i]))/65535;
  }else{  //Image data is float
    printf("Image depth %d is not recognized as integer type (%d or %d), Image data is treated as floats\n", imagein.depth(), CV_8U, CV_16U);
    imagein.addref();
    data = (Real*)imagein.data;
    ccmemMngr.registerMem(data, row*col*sizeof(Real));
  }
  return data;
}

void writeComplexImage(const char* name, void* data, int row, int column){
    FileStorage fs(name,FileStorage::WRITE);
    Mat output(row, column, float_cv_format(2));
    auto tmp = output.data;
    output.data = (uchar*)data;
    fs<<"data"<<output;
    fs.release();
    output.data = tmp;
}

void *readComplexImage(const char* name){
  FileStorage fs(name,FileStorage::READ);
  Mat image;
  fs["data"]>>(image);
  fs.release();
  size_t sz = image.rows*image.cols*sizeof(Real)*2;
  void *data = ccmemMngr.borrowCache(sz);
  memcpy(data, image.data, sz);
  return data;
};

