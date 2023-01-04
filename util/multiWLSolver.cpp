#include <iostream>
#include "imageReader.h"
#include <vector>
#include "common.h"
#include "fftw.h"
#include <fstream>

using namespace std;
double gaussian(double r2, double sigma){
	return exp(-r2/2/pow(sigma,2));
}

double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return gaussian(r2, sigma);
}

double gaussian_norm(double x, double y, double sigma){
  return 1./(2*M_PI*sigma*sigma)*gaussian(x,y,sigma);
}

double interpolate(double x, std::vector<double> &v){
	int integer = (int)x;
	double decimal = x-integer;
	if(integer>=v.size()) return v.back();
	if(decimal<1e-5) return v.at(integer);
	return v.at(integer)*(1-decimal)+v.at(integer+1)*decimal;
}

int main(int argc, char** argv){
	int range=1000;
	ofstream file1,file2,file3;
        file1.open("originalFunc.txt",ios::out);
        file2.open("mlFunc.txt",ios::out);
        file3.open("guessFunc.txt",ios::out);
	vector<double> originalFunc;
	vector<double> mlFunc;
	vector<double> lengths = {0.8,1,1.3};
	vector<double> weights = {1.0,1.,0.6};
	originalFunc.resize(range,0);
	mlFunc.resize(range,0);
	for(int i = 0; i < range; i++){
		originalFunc[i] = (gaussian(pow(i-300.,2), 50) + gaussian(pow(i-600.,2), 50) );
		file1<< originalFunc[i] << endl;
	}
	for(int i = 0; i < range; i++){
		for(int j = 0; j < lengths.size(); j++){
			if(i*lengths[j]>=0&&i*lengths[j]<range-1)
				mlFunc[i] += weights[j]*interpolate(i*lengths[j],originalFunc);
		}
		file2<< mlFunc[i] << endl;
	}
	vector<double> guessFunc[2];// = {originalFunc,originalFunc};
	guessFunc[0].resize(range,0);
	guessFunc[1].resize(range,0);
	double stepsize = 0.005;
	int niteration = 3000;
	double x = 0;
	for(int iter = 0; iter < niteration; iter++){
		for(int i = 0; i < range; i++){
			guessFunc[iter%2][i] = mlFunc[i];
			for(int j = 0; j < lengths.size(); j++){
				if(j==1) continue;
				if(i*lengths[j]>=0&&i*lengths[j]<range-1)
					guessFunc[iter%2][i]-=weights[j]*interpolate(i*lengths[j],guessFunc[!(iter%2)]);
			}
			if(guessFunc[iter%2][i]<0) guessFunc[iter%2][i]=0;
			guessFunc[iter%2][i]/=weights[1];
			guessFunc[iter%2][i] = guessFunc[iter%2][i]*stepsize + guessFunc[!(iter%2)][i]*(1-stepsize);
		}
	}
	for(int i = 0; i < range; i++){
		file3<<guessFunc[!(niteration%2)][i]<<endl;
	}
	
	
	return 0;
}

