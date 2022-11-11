#include "cuPlotter.h"
#include "cudaDefs.h"

void cuPlotter::initcuData(size_t sz){
  if(cuCache_data) cudaFree(cuCache_data);
  cudaMalloc(&cuCache_data, sz);
}

void cuPlotter::freeCuda(){
  cudaFree(cuCache_data);
}

__device__ Real cugetVal(mode m, complexFormat &data){
  if(m==IMAG) return data.y;
  if(m==MOD) return cuCabsf(data);
  if(m==MOD2) return data.x*data.x+data.y*data.y;
  if(m==PHASE){
    if(cuCabsf(data)==0) return 0;
    return atan(data.y/data.x);
  }
  return data.x;
}
__device__ Real cugetVal(mode m, Real &data){
  if(m==MOD2) return data*data;
  return data;
}

template <typename T>
__global__ void process(void* cudaData, pixeltype* cache, mode m, bool isFrequency=0, Real decay = 1, bool islog = 0){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int halfrow = cuda_row>>1;
  int halfcol = cuda_column>>1;
  int targetx = x;
  int targety = y;
  if(isFrequency) {
    targetx = x<halfrow?x+halfrow:(x-halfrow);
    targety = y<halfcol?y+halfcol:(y-halfcol);
  }
  T data = ((T*)cudaData)[index];
  Real target = decay*cugetVal(m,data);
  if(target < 0) target = 0;
  if(islog){
    if(target!=0)
      target = log2f(target)*cuda_rcolor/log2f(cuda_rcolor)+cuda_rcolor;
  }else target*=cuda_rcolor;
  if(target>=cuda_rcolor) {
    target=cuda_rcolor-1;
  }
  cache[targetx*cuda_column+targety] = floor(target);
}

void cuPlotter::processFloatData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog){
  process<Real><<<numBlocks, threadsPerBlock>>>(cudaData, cuCache_data, m, isFrequency, decay, islog);
  cudaMemcpy(cv_data, cuCache_data,rows*cols*sizeof(pixeltype), cudaMemcpyDeviceToHost); 
};
void cuPlotter::processComplexData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog){
  process<complexFormat><<<numBlocks, threadsPerBlock>>>(cudaData, cuCache_data, m,isFrequency, decay, islog);
  cudaMemcpy(cv_data, cuCache_data,rows*cols*sizeof(pixeltype), cudaMemcpyDeviceToHost); 
};
