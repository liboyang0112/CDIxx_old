#include <iostream>
#include <complex>
using namespace std;
const double pi = 3.141592653;

int main(int argc, char** argv){
	//GPU implementation: calcElement<<<x,y>>>(shiftx, shifty), return a x by y matrix
	//if(xmid*scale+shiftx <= row || ymid+shifty <= col) map{(xmid*scale+shiftx)*cols+(ymid+shifty)*scale, x*cols+y)} += return_value[x*cols+y]
	int N = 300;
	int x = 0;
	int y = 0;
	int p = 150;

	int xmid = round(x*double((N+1)/2+p)/N*2);
	int ymid = round(y*double((N+1)/2+p)/N*2);
	complex<double> tot = 0.;
	double tot1 = 0.;
	int range = 5;
	for(int xprime = xmid-range; xprime < xmid+range+1; xprime ++){
	for(int yprime = ymid-range; yprime < ymid+range+1; yprime ++){
	//Mat transformed(N, N, CV_8UC1);
	double k1 = 2*pi*(double(xprime+N/2+p)/(N+2*p)-double(x+N/2)/N);
	double k2 = 2*pi*(double(yprime+N/2+p)/(N+2*p)-double(y+N/2)/N);
	double mk1 = 2*pi*(double(xprime+N/2+p)/(N+2*p)-double(-x+N/2)/N);
	double mk2 = 2*pi*(double(yprime+N/2+p)/(N+2*p)-double(-y+N/2)/N);
	complex<double> sum = 0;
	double sum1 = 0;
	//for(int k = -p-N/2; k < (N+1)/2+p; k++){
	//	for(int l = -p-N/2; l < (N+1)/2+p; l++){
	for(int k = -N/2; k < (N+1)/2; k++){
		for(int l = -N/2; l < (N+1)/2; l++){
			//sum += exp(complex<double>(0, k*k1+l*k2)) + exp(complex<double>(0, k*mk1+l*mk2));
			sum1 += cos(k*k1+l*k2);// + cos(k*mk1+l*mk2);
		}
	}
	if(fabs(sum1/pow(N+2*p,2)) > 1e-10) cout<<"("<< xprime << "," << yprime<< ")="<<sum1/pow(N+2*p,2) << ", ";
	//cout<<"element="<<sum/pow(N+2*p,2)<<", "<<sum1/pow(N+2*p,2) << ", ";
	//tot+=sum;
	tot1+=sum1;
	//cout<<"element="<<sum<<", ";
	}
	cout<<endl;
	}
	cout<<tot1/pow(N+2*p,2)<<endl;
	return 0;
}

