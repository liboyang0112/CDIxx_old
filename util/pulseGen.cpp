#include <iostream>
#include <vector>
#include <common.h>

double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

double gaussian_norm(double x, double y, double sigma){
  return 1./(2*pi*sigma*sigma)*gaussian(x,y,sigma);
}

int main(int argc, char* argv){
	fftw_complex *freqDom, *timeDom;
	int Npoints = 1000; //Npoints = n_pixel*(lambdamax/lambdamin*-1)
	freqDom = (fftw_complex*)malloc(Npoints*sizeof(fftw_complex));
	timeDom = (fftw_complex*)malloc(Npoints*sizeof(fftw_complex));
}

Mat multiWLGen(Mat* original, double m){ //original image, ratio between long lambda and short lambda.
	int startx = original->rows*(1-1./m)/2;
	int starty = original->cols*(1-1./m)/2;
	return (*original)(Range(startx, original->rows-startx),Range(starty, original->cols-starty));
}

Mat* interpolation(Mat* lambdas, Mat* lambdal, double m, double n){ 
	//parameters: lambda short and lambda long, ratio between long lambda and short lambda, place of the interpolation point.
	//inside frequency range of short lambda image, the interpolation is from long lambda, otherwise from short lambda
	//interpolation only applies to lambda=[lambdas,lambdal]

}
