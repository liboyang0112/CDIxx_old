#include <iostream>
#include "imageReader.h"
#include <vector>
#include "common.h"
#include "fftw.h"

Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*pi*sigma*sigma)*gaussian(x,y,sigma);
}

int main(int argc, char** argv){
	/*
	fftw_format *freqDom, *timeDom;
	int Npoints = 1000; //Npoints = n_pixel*(lambdamax/lambdamin*-1)
	freqDom = (fftw_format*)malloc(Npoints*sizeof(fftw_format));
	timeDom = (fftw_format*)malloc(Npoints*sizeof(fftw_format));
	*/
	Mat input = readImage(argv[1]);
	int os = 2;
	Mat *img = extend(input,os);
	Mat *imgf = convertFromIntegerToComplex(*img);
	Mat *inputimg = convertFromComplexToInteger(imgf, 0, MOD2, 0, 1, "input", 0);
	imwrite("mergedinput.png",*inputimg);
	Real m = 2;
	Real dphaselambda = 10*pi;  // simulate lambda ~ lambda/m
	int ntime = 30;
	Real dphase = dphaselambda/ntime;
	/*
	for(int i = -ntime ; i < ntime; i++){
		Mat *merged = multiWLGen(imgf,0,m,1,dphase*i);
		convertFromComplexToInteger(merged, logged, MOD2, 1, 1, "merged", 1);
		plotColor(("merged"+to_string(i+ntime)+".png").c_str(),logged);
		delete merged;
	}
	*/
	Mat *merged = multiWLGenAVG_AC_FFT(imgf,0,m,0.51);
	//Mat *merged = multiWLGenAVG_AC_FFT(imgf,0,m,0.51);
	Mat *mergedmag = new Mat(merged->rows,merged->cols,float_cv_format(1));
	auto f = [&](int x, int y, Real &out, complex<Real> &in){
		out = abs(in);
	};
	imageLoop<decltype(f), Real, complex<Real>>(mergedmag, merged, &f);
	Mat *mergedMag = convertFO<Real>(mergedmag);
	imwrite("mergedavg_float.tiff",*mergedMag);
	Mat *tmp = convertFromComplexToInteger(merged, 0, MOD, 0, 1./pow(1.5,2), "merged", 0);
	//Mat *tmp = convertFromComplexToInteger<fftw_format>(merged, 0, MOD, 0, 100, "merged", 0);
	//plotColor("mergedavg.png",*tmp);
	imwrite("mergedavg.png",*tmp);
	delete tmp;
	tmp = convertFromComplexToInteger(merged, 0,MOD, 0, 1, "merged", 1);
	plotColor("mergedlog.png",tmp);
	delete tmp;
	Mat *autocorrelation = fftw(merged,0,1);
	tmp = convertFromComplexToInteger(autocorrelation, 0, REAL, 0, 1, "merged", 1);
	imwrite("mergedAC.png",*tmp);
	delete tmp;
	return 0;
}

