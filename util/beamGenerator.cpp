#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "format.h"
#include <random>
using namespace cv;
using namespace std;
//auto format_cv = CV_16UC(1);
auto format_cv = CV_64FC(1);
//auto format_cv = CV_8UC(1);
double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return exp(-r2/pow(sigma,2));
}
int main(int argc, char** argv )
{
  int row = 256;
  int column = 256;
  Mat image (row, column, float_cv_format(1), Scalar::all(0));
  int rad = row/2;
  int radcut = row/2;
  auto seed = (unsigned)time(NULL);
  srand(seed);
  Real* rowo;
  //char* rowo;
  double tot = 0;
  double totx = 0;
  double toty = 0;
  double sumx = 0;
  double sumy = 0;
  double max = 0;
  for(int x = 0; x < row ; x++){
    rowo =   image.ptr<Real>(x);
    //rowo = image.ptr<char>(x);
    for(int y = 0; y<column; y++){
      double r = hypot(x-row/2,y-row/2);
      //if(abs(x-row/2)<=1) continue;
      if(r<radcut) {
        Real randm = static_cast<Real>(rand())/RAND_MAX;
        //if(randm < 0.3) continue;
        rowo[y] = gaussian(x-row/2,y-row/2,rad);
      }
      //if(r<60) rowo[y] *= 0.4;
    }
  }

  imwrite("image.tiff", image);
  return 0;
}
