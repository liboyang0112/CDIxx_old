#include <iostream>
#include "matrixGen.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const dim3 threadsPerBlock(16,16);
__device__ __constant__ int cuda_row;
__device__ __constant__ int cuda_column;

__global__ void calcElement(double* matrixEle, int shiftx, int shifty, int paddingx, int paddingy){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  //shift to middle
  x-=cuda_row/2;
  y-=cuda_column/2;
  int xprime = round(x*double((cuda_row+1)/2+paddingx)/cuda_row*2)+shifty;
  int yprime = round(y*double((cuda_column+1)/2+paddingy)/cuda_column*2)+shiftx;
  if( xprime >= cuda_row || yprime >= cuda_column) return;
  double k1 = 2*M_PI*(double(xprime+cuda_row/2+paddingx)/(cuda_row+2*paddingx)-double(x+cuda_row/2)/cuda_row);
  double k2 = 2*M_PI*(double(yprime+cuda_column/2+paddingy)/(cuda_column+2*paddingy)-double(y+cuda_column/2)/cuda_column);
  double sum = 0;
  for(int k = -cuda_row/2; k < (cuda_row+1)/2; k++){
    for(int l = -cuda_column/2; l < (cuda_column+1)/2; l++){
      sum += cos(k*k1+l*k2);
    }
  }
	//printf("GPU calc, %d,%d, %f\n",cuda_row, cuda_column,sum);
  matrixEle[x*cuda_column + y] = sum;
}
void matrixGen(Sparse *matrix, int rows, int cols, int paddingx, int paddingy){
  dim3 numBlocks((rows-1)/threadsPerBlock.x+1, (cols-1)/threadsPerBlock.y+1);
  printf("<<<{%d,%d},{%d,%d}>>>\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
  cudaMemcpyToSymbol(cuda_column,&cols,sizeof(cols));
  cudaMemcpyToSymbol(cuda_row,&rows,sizeof(rows));
  int widthx = 3;
  int widthy = 3;
  double *cuda_matrix;
  double *matrixEle;
  size_t sz = sizeof(double)*rows*cols;
  matrixEle = (double*)malloc(sz);
  gpuErrchk(cudaMalloc((void**)&cuda_matrix, sz));
  for(int shiftx = -widthx; shiftx <= widthx; shiftx++){
    for(int shifty = -widthy; shifty <= widthy; shifty++){
      calcElement<<<numBlocks,threadsPerBlock>>>(cuda_matrix, shiftx, shifty, paddingx, paddingy);
      cudaMemcpy(matrixEle, cuda_matrix, sz, cudaMemcpyDeviceToHost);
      for(int index = 0; index < rows*cols; index ++){
        int x = index/cols;
        int y = index%cols;
        int xprime = round(x*double((rows+1)/2+paddingx)/rows*2)+shifty;
        int yprime = round(y*double((cols+1)/2+paddingy)/cols*2)+shiftx;
        if(matrixEle[index] > 1e-8){
          int cord[2] = {xprime*cols + yprime, index};
	  (*matrix)[cord] += matrixEle[index];
	}
      }
    }
  }
}
