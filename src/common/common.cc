#include "imageReader.h"
#include "readCXI.h"
#include "common.h"
#include "fstream"
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

void getNormSpectrum(const char* fspectrum, const char* ccd_response, Real &startLambda, int &nlambda, Real *& outlambda, Real *& outspectrum){
  std::vector<Real> spectrum_lambda;
  std::vector<Real> spectrum;
  std::vector<Real> ccd_lambda;
  std::vector<Real> ccd_rate;
  std::ifstream file_spectrum, file_ccd_response;
  Real threshold = 5e-3;
  file_spectrum.open(fspectrum);
  file_ccd_response.open(ccd_response);
  Real lambda, val, maxval;
  maxval = 0;
  while(file_spectrum){
    file_spectrum >> lambda >> val;
    spectrum_lambda.push_back(lambda);
    spectrum.push_back(val);
    if(val > maxval) maxval = val;
  }
  while(file_ccd_response){
    file_ccd_response >> lambda >> val;
    ccd_lambda.push_back(lambda);
    ccd_rate.push_back(val);
  }
  Real endlambda = ccd_lambda.back();
  bool isShortest = 1;
  int ccd_n = 0;
  nlambda = 0;
  for(int i = 0; i < spectrum.size(); i++){
    if(spectrum_lambda[i] < ccd_lambda[0] || spectrum_lambda[i]<startLambda) continue;
    if(spectrum_lambda[i] >= endlambda) break;
    if(isShortest && spectrum[i] < threshold*maxval) continue;
    if(isShortest) startLambda = spectrum_lambda[i];
    isShortest = 0;
    spectrum_lambda[nlambda] = spectrum_lambda[i]/startLambda;
    while(ccd_lambda[ccd_n] < spectrum_lambda[i]) ccd_n++;
    Real dx = (spectrum_lambda[i]-ccd_lambda[ccd_n-1])/(ccd_lambda[ccd_n] - ccd_lambda[ccd_n-1]);
    Real ccd_rate_i = ccd_rate[ccd_n-1]*(1-dx) + ccd_rate[ccd_n]*dx;
    //printf("ccd_rate = %f , dx = %f, spectrum = %f\n", ccd_rate_i, dx, spectrum[nlambda]);
    spectrum[nlambda] = spectrum[i]*ccd_rate_i/maxval;
    nlambda++;
  }
  outlambda = (Real*) ccmemMngr.borrowCache(sizeof(Real)*nlambda);
  outspectrum = (Real*) ccmemMngr.borrowCache(sizeof(Real)*nlambda);
  for(int i = 0; i < nlambda; i++){
    outlambda[i] = spectrum_lambda[i];
    outspectrum[i] = spectrum[i];
  }
}
