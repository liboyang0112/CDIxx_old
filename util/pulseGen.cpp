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
	/*
	fftw_complex *freqDom, *timeDom;
	int Npoints = 1000; //Npoints = n_pixel*(lambdamax/lambdamin*-1)
	freqDom = (fftw_complex*)malloc(Npoints*sizeof(fftw_complex));
	timeDom = (fftw_complex*)malloc(Npoints*sizeof(fftw_complex));
	*/
	Mat input = readImage(argv[1]);
	Mat *img = extend(input,4);
	Mat *imgf = convertFromIntegerToComplex(*img);
	Mat *inputimg = convertFromComplexToInteger(imgf, 0, MOD2, 0, 1, "input", 0);
	imwrite("mergedinput.png",*inputimg);
	double m = 2;
	double dphaselambda = 10*pi;  // simulate lambda ~ lambda/m
	int ntime = 30;
	double dphase = dphaselambda/ntime;
	Mat *logged = new Mat(img->rows, img->cols, CV_16UC1);
	/*
	for(int i = -ntime ; i < ntime; i++){
		Mat *merged = multiWLGen(imgf,0,m,1,dphase*i);
		convertFromComplexToInteger(merged, logged, MOD2, 1, 1, "merged", 1);
		plotColor(("merged"+to_string(i+ntime)+".png").c_str(),logged);
		delete merged;
	}
	*/
	Mat *merged = multiWLGenAVG_MAT_FFT(imgf,0,m,0.51);
	Mat *mergedmag = new Mat(merged->rows,merged->cols,CV_64FC1);
	auto f = [&](int x, int y, double &out, complex<double> &in){
		out = abs(in);
	};
	imageLoop<decltype(f), double, complex<double>>(mergedmag, merged, &f);
	Mat *mergedMag = convertFO<double>(mergedmag);
	imwrite("mergedavg_float.tiff",*mergedMag);
	convertFromComplexToInteger(merged, logged, MOD, 0, 1, "merged", 1);
	plotColor("mergedavg.png",logged);
	//imwrite("mergedavg.png",*logged);
	Mat *autocorrelation = fftw(merged,0,1);
	convertFromComplexToInteger(autocorrelation, logged, MOD, 0, 1, "merged", 1);
	imwrite("mergedAC.png",*logged);
	return 0;
}

