#include <iostream>
#include "mergePixel.h"

__global__ void mergePixel(pixeltype* src, pixeltype* bkg, Real* output, char* oelabel, int mergeDepthx, int mergeDepthy){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= config->cuda_row/mergeDepthx || y >= config->cuda_column/mergeDepthy) return;
  int index=x*config->cuda_column + y;
  int indexP=x*config->cuda_column*mergeDepthx + y*mergeDepthy;
  int tmp = 0;
  oelabel[index] = 0;
  for(int i = 0; i < mergeDepthx; i++){
    for(int j = 0; j < mergeDepthy; j++){
      int ele = src[indexP+mergeDepthx*i+mergeDepthy];
      if(ele >= cuda_rcolor-1) oelabel[index] = 1;
      tmp += Real(ele);
    }
  }
  tmp/=mergeDepthx*mergeDepthy;
}
