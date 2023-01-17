#include <iostream>
#include "cudaDefs.h"
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

__global__ void calcElement(float* matrixEle, int shiftx, int shifty, int paddingx, int paddingy){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index=x*cuda_column + y;
  //shift to middle
  x-=cuda_row/2;
  y-=cuda_column/2;
  int xprime = round(x*float((cuda_row+1)/2+paddingx)/cuda_row*2)+shifty;
  int yprime = round(y*float((cuda_column+1)/2+paddingy)/cuda_column*2)+shiftx;
  if( abs(xprime) >= cuda_row/2 || abs(yprime) >= cuda_column/2) {
    matrixEle[index] = 0;
    return;
  }
  float k1 = 2*M_PI*(float(xprime+cuda_row/2+paddingx)/(cuda_row+2*paddingx)-float(x+cuda_row/2)/cuda_row);
  float k2 = 2*M_PI*(float(yprime+cuda_column/2+paddingy)/(cuda_column+2*paddingy)-float(y+cuda_column/2)/cuda_column);
  //float m1 = 2*M_PI*(float(xprime+cuda_row/2+paddingx)/(cuda_row+2*paddingx)+float(x+cuda_row/2)/cuda_row);
  //float m2 = 2*M_PI*(float(yprime+cuda_column/2+paddingy)/(cuda_column+2*paddingy)+float(y+cuda_column/2)/cuda_column);
  float sum = 0;
  float tmp;
  for(int k = -cuda_row/2; k < (cuda_row+1)/2; k++){
    tmp = k*k1+(-cuda_column/2)*k2;
    for(int l = -cuda_column/2; l < (cuda_column+1)/2; l++){
      sum += cos(tmp);
      tmp+=k2;
    }
  }
  //if(x==0&&y==0) printf("GPU calc, %d,%d, %f, %f, %f\n",x, y,k1,k2, sum/cuda_row/cuda_column);
  matrixEle[index] = sum/cuda_row/cuda_column;
}
void matrixGen(Sparse *matrix, int rows, int cols, int paddingx, int paddingy, float weight){
  dim3 numBlocks((rows-1)/threadsPerBlock.x+1, (cols-1)/threadsPerBlock.y+1);
  printf("<<<{%d,%d},{%d,%d}>>>\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
  cudaMemcpyToSymbol(cuda_column,&cols,sizeof(cols));
  cudaMemcpyToSymbol(cuda_row,&rows,sizeof(rows));
  int widthx = 2;
  int widthy = 2;
  float *cuda_matrix;
  float *matrixEle;
  size_t sz = sizeof(float)*rows*cols;
  matrixEle = (float*)malloc(sz);
  gpuErrchk(cudaMalloc((void**)&cuda_matrix, sz));
  for(int shiftx = -widthx; shiftx <= widthx; shiftx++){
    for(int shifty = -widthy; shifty <= widthy; shifty++){
      //if(abs(shiftx)>2 && abs(shifty)>2) continue;
      calcElement<<<numBlocks,threadsPerBlock>>>(cuda_matrix, shiftx, shifty, paddingx, paddingy);
      cudaMemcpy(matrixEle, cuda_matrix, sz, cudaMemcpyDeviceToHost);
      for(int index = 0; index < rows*cols; index ++){
        if(fabs(matrixEle[index]) > 1e-5){
          int x = index/cols-rows/2;
          int y = index%cols-cols/2;
	  //if(x == 10 && y == 10) printf("%f, ", matrixEle[index]);
          int xprime = round(x*float((rows+1)/2+paddingx)/rows*2)+shifty+rows/2;
          int yprime = round(y*float((cols+1)/2+paddingy)/cols*2)+shiftx+cols/2;
          int cord[2] = {xprime*cols + yprime, index};
	  (*matrix)[cord] += matrixEle[index]*weight;
	}
      }
    }
    printf("matrix size: %lu\n", matrix->matrix.size());
    //printf("\n");
  }
}
