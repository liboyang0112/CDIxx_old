#include <cuda.h>
#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "common.h"
#include <applyMod.h>

__global__ void applyMod(cufftDoubleComplex* source, cufftDoubleComplex* target, int row, int column, support *bs = 0){
  assert(source!=0);
  assert(target!=0);
  double tolerance = 0.5/rcolor*scale;
  double maximum = pow(mergeDepth,2)*scale;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  index = x + y*row;
  if(bs!=0){
    int tx = x;
    if(x >= row/2) tx -= row/2;
    else tx += row/2;
    int ty = y;
    if(y >= column/2) ty -= column/2;
    else ty += column/2;
    if(bs->isInside(tx,ty)) {
      continue;
    }
  }
  cufftDoubleComplex &targetdata = target[index];
  cufftDoubleComplex &sourcedata = source[index];
  double ratio = 1;
  double mod2 = targetdata.x*targetdata.x + targetdata.y*targetdata.y;
  double srcmod2 = sourcedata.x*sourcedata.x + sourcedata.y*sourcedata.y;
  if(mod2>=maximum) {
    mod2 = max(maximum,srcmod2);
  }
  if(srcmod2 == 0){
    sourcedata.x = sqrt(mod2);
    sourcedata.y = 0;
    continue;
  }
  double diff = mod2-srcmod2;
  if(diff>tolerance){
    ratio = sqrt((mod2-tolerance)/srcmod2);
  }else if(diff < -tolerance ){
    ratio = sqrt((mod2+tolerance)/srcmod2);
  }
  sourcedata.x *= ratio;
  sourcedata.y *= ratio;
}
