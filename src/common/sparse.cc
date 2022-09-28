#include "sparse.h"

Sparse::Sparse(int rows_, int cols_) : rows(rows_), cols(cols_){}

float& Sparse::operator[] (int x[2]){
	uint64_t index = x[0];
	index = index*cols+x[1];
	return matrix[index];
}	
float Sparse::operator() (int x, int y){
	uint64_t index = x;
	index = index*static_cast<uint64_t>(cols)+y;
	auto iter = matrix.find(index);
	if(iter == matrix.end()) return 0;
	return iter->second;
}	
float* Sparse::operator* (float *X){
	float *ret = (float*)malloc(rows*cols*sizeof(float));
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

