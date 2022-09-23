#ifndef __SPARSE__
#define __SPARSE__
#include <map>
#include <iostream>

class Sparse{
public:
	Sparse(int rows_, int cols_);
	const uint64_t rows;
	const uint64_t cols;
	std::map<uint64_t, double> matrix;
	double &operator[] (int x[2]);
	double operator() (int x, int y);
	double* operator* (double *x);
	template<typename T>
	T* operator* (T *X){
	        T *ret = (T*)malloc(cols*sizeof(T));
	        for(uint64_t i = 0; i < cols; i++) ret[i] = 0;
	        for(auto iter : matrix){
	                uint64_t index = iter.first;
	                int x = index/cols;
	                int y = index%cols;
	                ret[x] += iter.second*X[y];
	        }
	        return ret;
	}
	void convertToCholmod();
	void saveToFile();
	void readFromFile();
};
#endif
