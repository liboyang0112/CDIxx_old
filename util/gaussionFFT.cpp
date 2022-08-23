# include <stdlib.h>
# include <cassert>
# include <stdio.h>
# include <time.h>
# include <random>
# include <complex.h>

#include <stdio.h>
#include <opencv2/opencv.hpp>
# include <fftw3.h>
using namespace std;

void test01 ( void );
double frand ( void ){
	return rand()%1000;
};
const double pi = 3.1415927;
using namespace cv;
double gaussian(double x, double y, double sigma){
    double r2 = pow(x,2) + pow(y,2);
    return 1./sqrt(2*pi*sigma)*exp(-r2/2/pow(sigma,2));
}

enum mode {MOD2,MOD, REAL, IMAG, PHASE};
/******************************************************************************/

fftw_complex* fftw ( fftw_complex* in, int row, int column, fftw_complex *out = 0, bool isforward = 1)
{
  ;
  fftw_plan plan_forward;
  double ratio = 1./sqrt(row*column);
  if(out == 0) out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );

  plan_forward = fftw_plan_dft_2d ( row, column,  in, out, isforward?FFTW_FORWARD:FFTW_BACKWARD, FFTW_ESTIMATE );

  fftw_execute ( plan_forward );

  for(int i = 0; i < row*column ; i++){
    out[i][0]*=ratio;
    out[i][1]*=ratio;
  }
/*only used to check the correctness of the transformation
  fftw_plan plan_backward;
  plan_backward = fftw_plan_dft_2d ( row, column, out, in, FFTW_BACKWARD, FFTW_ESTIMATE );

  fftw_execute ( plan_backward );

  for(int i = 0; i < row*column ; i++){
    in[i][0]*=ratio;
    in[i][1]*=ratio;
  }
  fftw_destroy_plan ( plan_backward );
*/
  fftw_destroy_plan ( plan_forward );

  return out;
}
/******************************************************************************/

fftw_complex* convertFromOpencvToFFTW(Mat image, int row, int column, fftw_complex* cache = 0){
  if(!cache) cache = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
  uchar* rowp;
  for(int x = 0; x < row ; x++){
    rowp = image.ptr<uchar>(x);
    for(int y = 0; y<column; y++){
      cache[x*column+y][0] = (double)rowp[y];
      cache[x*column+y][1] = 0;
    }
  }
  return cache;
}

void convertFromFFTWToOpencv(Mat &image, int row, int column, fftw_complex* cache, mode m, bool isFrequency, double decay = 1){
  uchar* rowp;
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = image.ptr<uchar>(targetx);
    for(int y = 0; y<column; y++){
      double target;
      complex<double> tmpc(cache[x*column+y][0],cache[x*column+y][1]);
      switch(m){
        case MOD:
          target = std::abs(tmpc);
          break;
        case MOD2:
          target = pow(std::abs(tmpc),2);
          break;
        case REAL:
          target = cache[x*column+y][0];
          break;
        case IMAG:
          target = cache[x*column+y][1];
          break;
	case PHASE:
	  target = std::arg(tmpc)/2/pi;
	default:
	  target = cache[x*column+y][0];
      }
      target*=255*decay;
      if(target<0) target = -target;
      if(target>255) target=255;
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      rowp[targety] = (int)target;
    }
  }
}
void applyPhase(fftw_complex* source, fftw_complex* target, int row, int column){
  assert(source!=0);
  assert(target!=0);
  for(int i = 0; i < row*column ; i++){
    double mod = hypot(target[i][0],target[i][1]);
    double phase;
    complex<double> tmpc(source[i][0],source[i][1]);
    phase = std::arg(tmpc);
    target[i][0] = cos(phase)*mod;
    target[i][1] = sin(phase)*mod;
  }
}
int main(int argc, char** argv )
{
    int row = 3000;
    int column = 3000;
    Mat image (row, column, CV_8UC(1), Scalar::all(0));
    fftw_complex* inputField = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
    fftw_complex* targetField = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
    for(int x = 0; x < row ; x++){
      for(int y = 0; y<column; y++){
	double Emod =  3*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.1);
	double Emodt = 3*gaussian(((double)x)/row-0.1,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.2,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.3,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.4,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.6,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.7,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.8,((double)y)/column-0.5,0.01)
		     + 3*gaussian(((double)x)/row-0.9,((double)y)/column-0.5,0.01);
	double randphase = static_cast<double>(rand())/RAND_MAX*2*pi;
        inputField[x*column+y][0] = Emod;//*sin(randphase);
        inputField[x*column+y][1] = 0;//Emod*cos(randphase);
        int targetx = x<row/2?x+row/2:(x-row/2);
        int targety = y<column/2?y+column/2:(y-column/2);
        targetField[targetx*column+targety][0] = Emodt*sin(randphase);
        targetField[targetx*column+targety][1] = Emodt*cos(randphase);
      }
    }
    fftw_complex* fftresult = 0;
    fftw_complex* fftbwresult = 0;
    convertFromFFTWToOpencv(image,row, column, inputField, MOD2,0);
    imwrite("inputField.png",image);
    convertFromFFTWToOpencv(image,row, column, targetField, MOD2,1);
    imwrite("targetField.png",image);
    for(int i = 0; i<1; i++){
      fftresult = fftw(inputField, row, column,fftresult,1);
      applyPhase(fftresult,targetField,row,column);
      fftbwresult = fftw(targetField,row,column,fftbwresult,0);
      applyPhase(fftbwresult,inputField,row,column);
    }
    convertFromFFTWToOpencv(image,row, column, fftresult, MOD2,1,0.05);
    imwrite("fft.png",image);
    convertFromFFTWToOpencv(image,row, column, fftbwresult, MOD2,0);
    imwrite("fftbw.png",image);
    convertFromFFTWToOpencv(image,row, column, fftbwresult, PHASE,0);
    imwrite("phase.png",image);


    //waitKey(0);
    return 0;
}
