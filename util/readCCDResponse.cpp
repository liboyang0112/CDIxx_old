#include <stdio.h>
#include <fftw3.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "format.h"
#include "imageReader.h"
using namespace cv;
using namespace std;

bool onCurve(unsigned char *pixColor, unsigned char *curveColor, int tolerance){
  bool retval = 1;
  for ( int i = 0; i < 3; i++){
    if(abs(pixColor[i]-curveColor[i]) > tolerance) {
      retval = 0;
      break;
    }
  }
  return retval;
}

int main(int argc, char** argv )
{
  int start[] = {126,384};
  int end[] = {676,51};
  Real startlambda = 400;
  Real endlambda = 1000;
  int tolerance = 50;
  unsigned char curveColor[3] = {200, 150, 40};

  Mat imagein = imread( argv[1], IMREAD_UNCHANGED );

  int row = imagein.rows;
  int column = imagein.cols;
  printf("image %d x %d\n", row, column);
  int nlambda = end[0]-start[0];
  Real *lambdas = (Real*) malloc(nlambda*sizeof(Real));
  Real *rate = (Real*) malloc(nlambda*sizeof(Real));
  int *count = (int*) malloc(nlambda*sizeof(int));
  memset(count, 0, nlambda*sizeof(int));
  memset(rate, 0, nlambda*sizeof(Real));

  cv::Vec3b* rowp;
  for(int x = start[1]; x>end[1]; x--){
    rowp = imagein.ptr<cv::Vec3b>(x);
    for(int y = start[0]; y < end[0] ; y++){
    if(x == start[1]) lambdas[y-start[0]] = startlambda + (endlambda-startlambda)/(end[0]-start[0])*(y-start[0]);
      if(onCurve(&rowp[y][0], curveColor, tolerance)) {
        Real tmp = Real(start[1]-x)/(start[1]-end[1]);
        printf("find curve at (%d, %d) = [%d, %d, %d], rate= %f\n", x, y, rowp[y][0], rowp[y][1], rowp[y][2], tmp);
        rowp[y][2] = 255;
        rowp[y][0] = rowp[y][1] = 0;
        rate[y-start[0]] += tmp;
        count[y-start[0]] += 1;
      }
      //else printf("find non-curve at (%d, %d)=[%d, %d, %d]\n", x, y, rowp[y][0], rowp[y][1], rowp[y][2]);
    }
  }
  ofstream outfile;
  outfile.open("ccd_response.txt", ios::out);
  for(int x = 0; x < nlambda; x++){
    if(count[x]) rate[x]/=count[x];
    outfile << lambdas[x] << " "<< rate[x] << endl;
  }
  outfile.close();
  imwrite("out.png", imagein);
  return 0;
}
