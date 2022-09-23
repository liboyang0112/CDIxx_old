#include "sparse.h"

Sparse::Sparse(int rows_, int cols_) : rows(rows_), cols(cols_){}

double& Sparse::operator[] (int x[2]){
	uint64_t index = x[0];
	index = index*cols+x[1];
	return matrix[index];
	/*
	auto iter = matrix.find(index);
	if(iter == matrix.end()) return matrix.insert({index,0}).second;
	printf("%p=%lu\n",&(iter->second),iter->first);
	return &iter->second;
	*/
}	
double Sparse::operator() (int x, int y){
	uint64_t index = x;
	index = index*static_cast<uint64_t>(cols)+y;
	auto iter = matrix.find(index);
	if(iter == matrix.end()) return 0;
	return iter->second;
}	
double* Sparse::operator* (double *X){
	double *ret = (double*)malloc(rows*cols*sizeof(double));
	for(uint64_t i = 0; i < rows*cols; i++) ret[i] = 0;
	for(auto iter : matrix){
		uint64_t index = iter.first;
		int x = index/cols;
		int y = index%cols;
		ret[y] += iter.second*X[x];
	}
	return ret;
}
void Sparse::saveToFile(){}
void Sparse::convertToCholmod(){}
void Sparse::readFromFile(){}

