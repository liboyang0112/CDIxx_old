#include "cuPlotter.h"
#include "cudaDefs.h"
#include <cassert>

void cuPlotter::initcuData(size_t sz){
  if(cuCache_data) memMngr.returnCache(cuCache_data);
  if(cuCache_float_data) memMngr.returnCache(cuCache_float_data);
  if(cuCache_complex_data) memMngr.returnCache(cuCache_complex_data);
  cuCache_data = (pixeltype*) memMngr.borrowCache(sz*sizeof(pixeltype));
  cuCache_float_data = (Real*) memMngr.borrowCache(sz*sizeof(Real));
  cuCache_complex_data = (complexFormat*) memMngr.borrowCache(sz*sizeof(complexFormat));
}

void cuPlotter::freeCuda(){
  if(cuCache_data) memMngr.returnCache(cuCache_data);
  if(cuCache_float_data) memMngr.returnCache(cuCache_float_data);
  if(cuCache_complex_data) memMngr.returnCache(cuCache_complex_data);
}

__device__ Real cugetVal(mode m, complexFormat &data){
  if(m==IMAG) return data.y;
  if(m==MOD) return cuCabsf(data);
  if(m==MOD2) return data.x*data.x+data.y*data.y;
  if(m==PHASE){
    return atan2(data.y,data.x)/2/M_PI+0.5;
  }
  if(m==PHASERAD){
    return atan2(data.y,data.x);
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
  if(target!=target) {
 //   printf("ERROR: target is NAN\n");
 //   exit(0);
  }
  cache[targetx*cuda_column+targety] = floor(target);
}
__global__ void getPhase(void* cudaData, Real* cache, mode m, bool isFrequency=0, Real decay = 1){
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
  cache[targetx*cuda_column+targety] =decay*cugetVal(m,((complexFormat*)cudaData)[index]);
}

void cuPlotter::processPhaseData(void* cudaData, const mode m, bool isFrequency, Real decay){
  cudaF(getPhase)(cudaData, (Real*)cuCache_float_data, m, isFrequency, decay);
  cudaMemcpy(cv_float_data, cuCache_float_data,rows*cols*sizeof(Real), cudaMemcpyDeviceToHost); 
};

void cuPlotter::saveFloatData(void* cudaData){
  cudaMemcpy(cv_float_data, cudaData, rows*cols*sizeof(Real), cudaMemcpyDeviceToHost); 
};
void cuPlotter::saveComplexData(void* cudaData){
  cudaMemcpy(cv_complex_data, cudaData, rows*cols*sizeof(complexFormat), cudaMemcpyDeviceToHost); 
};
void cuPlotter::processFloatData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog){
  cudaF(process<Real>)(cudaData, cuCache_data, m, isFrequency, decay, islog);
  cudaMemcpy(cv_data, cuCache_data,rows*cols*sizeof(pixeltype), cudaMemcpyDeviceToHost); 
};
void cuPlotter::processComplexData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog){
  cudaF(process<complexFormat>)(cudaData, cuCache_data, m,isFrequency, decay, islog);
  cudaMemcpy(cv_data, cuCache_data,rows*cols*sizeof(pixeltype), cudaMemcpyDeviceToHost); 
};
