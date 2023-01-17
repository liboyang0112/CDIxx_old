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
  return 1./(2*M_PI*sigma*sigma)*gaussian(x,y,sigma);
}

int main(int argc, char** argv){
	/*
	fftw_format *freqDom, *timeDom;
	int Npoints = 1000; //Npoints = n_pixel*(lambdamax/lambdamin*-1)
	freqDom = (fftw_format*)malloc(Npoints*sizeof(fftw_format));
	timeDom = (fftw_format*)malloc(Npoints*sizeof(fftw_format));
	*/
	int row, col;
	Real* inputdata = readImage(argv[1], row, col);
	Mat input(row,col,float_cv_format(1));

  memcpy(input.data, inputdata, input.total()*sizeof(Real));
	int os = 2;
	Mat *imgf = extend(input,os);
	Real m = 2;
	Real dphaselambda = 10*M_PI;  // simulate lambda ~ lambda/m
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
	auto f1 = [&](int x, int y, Real &out, complex<Real> &in){
		in = complex<Real>(sqrt(out), 0);
	};
  Mat *imgc = new Mat(row*os, col*os, float_cv_format(2));
	imageLoop<decltype(f1), Real, complex<Real>>(imgf, imgc, &f1);
	Mat *tmp = convertFromComplexToInteger(imgc, 0, MOD2, 0, 1, "merged", 0);
	imwrite("input.png",*tmp);
  
	Mat *merged = multiWLGenAVG_MAT(imgc,0,m,0);
	//Mat *merged = multiWLGenAVG_AC_FFT(imgf,0,m,0.51);
	//Mat *mergedmag = new Mat(merged->rows,merged->cols,float_cv_format(1));
	//auto f = [&](int x, int y, Real &out, complex<Real> &in){
	//	out = abs(in);
	//};
	//imageLoop<decltype(f), Real, complex<Real>>(mergedmag, merged, &f);
	//Mat *mergedMag = convertFO<Real>(mergedmag);
	//imwrite("mergedavg_float.tiff",*mergedMag);
	tmp = convertFromComplexToInteger(merged, tmp, MOD, 0, 0.5, "merged", 0);
	//Mat *tmp = convertFromComplexToInteger<fftw_format>(merged, 0, MOD, 0, 100, "merged", 0);
	//plotColor("mergedavg.png",*tmp);
	imwrite("mergedavg.png",*tmp);
	delete tmp;
	tmp = convertFromComplexToInteger(merged, 0,MOD, 0, 1, "merged", 1);
	plotColor("mergedlog.png",tmp);
	delete tmp;
	Mat *autocorrelation = fftw(merged,0,1);
	tmp = convertFromComplexToInteger(autocorrelation, 0, MOD, 1, 1, "merged", 1);
	plotColor("mergedAC.png",tmp);
	delete tmp;
	return 0;
}

