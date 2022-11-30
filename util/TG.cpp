#include <complex>
# include <cassert>
# include <stdio.h>
# include <time.h>
# include <random>

#include <stdio.h>
#include "fftw.h"
#include <iostream>
#include <fstream>
#include <libconfig.h++>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "imageReader.h"
#include <ctime>

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

// This code only apply to image with height and width in power of 2, i.e. ... 256, 512, 1024, .... due to Cuda restrictions.
//#define Bits 16
using namespace cv;
Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*M_PI*sigma*sigma)*gaussian(x,y,sigma);
}

/******************************************************************************/

int main(int argc, char** argv )
{
    int row = 512;
    int column = 512;
    //These are in mm;
    Real lambda = 800e-6;
    Real dhole = 2; // distance between two holes of pump light
    Real focus = 20;
    Real pixelsize = 3e-3;
    Real spotSize = 60e-3;
    Real dn = 1e-5;
    Real dx = 0.1;
    Real phi0 = dn*dx/lambda;

    int spotpix = spotSize/pixelsize;
    Real k = sin(dhole/2/focus)*2*M_PI/lambda * pixelsize;
    Mat image (row, column, CV_16UC(1), Scalar::all(0));
    Mat imageInput (row, column, float_cv_format(2), Scalar::all(0));
    Mat imageTarget (row, column, float_cv_format(2), Scalar::all(0));
    fftw_format* inputField = (fftw_format*) imageInput.data;
    for(int x = 0; x < row ; x++){
      for(int y = 0; y<column; y++){
	Real Emod =  gaussian(x-0.5*row,y-0.5*column,spotpix);
	/*
	Real Emodt = 3*gaussian(((Real)x)/row-0.1,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.2,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.3,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.4,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.5,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.6,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.7,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.8,((Real)y)/column-0.5,0.01)
		     + 3*gaussian(((Real)x)/row-0.9,((Real)y)/column-0.5,0.01);
		     */
	Real phase = cos(k * x)*phi0;
        inputField[x+y*row] = fftw_format(Emod*sin(phase),Emod*cos(phase));
      }
    }
    printf("doing fftw\n");
    fftw(&imageInput, &imageTarget, 1);
    convertFromComplexToInteger(&imageInput, &image, MOD2,0);
    imwrite("inputIntensity.png",image);
    convertFromComplexToInteger(&imageInput, &image, PHASE,0);
    imwrite("inputPhase.png",image);
    convertFromComplexToInteger(&imageTarget, &image, MOD2,1,1e6);
    imwrite("targetField.png",image);
    return 0;
}
