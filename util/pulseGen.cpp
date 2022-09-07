#include <iostream>
#include "imageReader.h"
#include <vector>
#include "common.h"
#include "fftw.h"

double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

double gaussian_norm(double x, double y, double sigma){
  return 1./(2*pi*sigma*sigma)*gaussian(x,y,sigma);
}

int main(int argc, char** argv){
	fftw_complex *freqDom, *timeDom;
	int Npoints = 1000; //Npoints = n_pixel*(lambdamax/lambdamin*-1)
	freqDom = (fftw_complex*)malloc(Npoints*sizeof(fftw_complex));
	timeDom = (fftw_complex*)malloc(Npoints*sizeof(fftw_complex));
	Mat input = readImage(argv[1]);
	Mat *img = extend(input,4);
	Mat *imgf = convertFromIntegerToComplex(*img);
	int m = 2;
	double dphaselambda = 10*pi;  // simulate lambda ~ lambda/m
	int ntime = 30;
	double dphase = dphaselambda/ntime;
	Mat *logged = new Mat(img->rows/m, img->cols/m, CV_16UC1);
	/*
	for(int i = -ntime ; i < ntime; i++){
		Mat *merged = multiWLGen(imgf,0,m,1,dphase*i);
		convertFromComplexToInteger(merged, logged, MOD2, 1, 1, "merged", 1);
		plotColor(("merged"+to_string(i+ntime)+".png").c_str(),logged);
		delete merged;
	}
	*/
	Mat *merged = multiWLGenAVG(imgf,0,m,1);
	convertFromComplexToInteger(merged, logged, MOD, 0, 1, "merged", 1);
	plotColor("mergedavg.png",logged);
	return 0;
}

