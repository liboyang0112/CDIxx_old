#include "cudaConfig.h"
#include "initCuda.h"
#include <iostream>
using namespace std;
void init_cuda_image(int rows, int cols, int rcolor, Real scale){
    cudaMemcpyToSymbol(cuda_row,&rows,sizeof(rows));
    cudaMemcpyToSymbol(cuda_column,&cols,sizeof(cols));
    Real ratio = 1./sqrt(rows*cols);
    cudaMemcpyToSymbol(cuda_rcolor,&rcolor,sizeof(rcolor));
    cudaMemcpyToSymbol(cuda_scale,&scale,sizeof(scale));
    numBlocks.x=(rows-1)/threadsPerBlock.x+1;
    numBlocks.y=(cols-1)/threadsPerBlock.y+1;
};
void init_fft(int rows, int cols){
    if(!plan){
      plan = new cufftHandle();
      planR2C = new cufftHandle();
    }else{
      cufftDestroy(*plan);
      cufftDestroy(*planR2C);
    }
    cufftPlan2d ( plan, rows, cols, FFTformat);
    cufftPlan2d ( planR2C, rows, cols, FFTformatR2C);
}
__global__ void fillRedundantR2C(complexFormat* data, complexFormat* dataout, Real factor){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int targetIndex = x*(cuda_column/2+1)+y;
  if(y <= cuda_column/2) {
    dataout[index].x = data[targetIndex].x*factor;
    dataout[index].y = data[targetIndex].y*factor;
    return;
  }
  if(x==0) {
      targetIndex = cuda_column-y;
  }else{
      targetIndex = (cuda_row-x)*(cuda_column/2+1)+cuda_column-y;
  }
  dataout[index].x = data[targetIndex].x*factor;
  dataout[index].y = -data[targetIndex].y*factor;
}

__global__ void applyNorm(complexFormat* data, Real factor){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  data[index].x*=factor;
  data[index].y*=factor;
}

__global__ void multiply(complexFormat* src, complexFormat* target){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  src[index] = cuCmulf(src[index], target[index]);
}
__global__ void forcePositive(complexFormat* a){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(a[index].x<0) a[index].x=0;
  a[index].y = 0;
}

__global__ void multiply(complexFormat* store, complexFormat* src, complexFormat* target){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  store[index] = cuCmulf(src[index], target[index]);
}

__global__ void extendToComplex(Real* a, complexFormat* b){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  b[index].x = a[index];
  b[index].y = 0;
}

__global__ void applyNorm(Real* data, Real factor){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  data[index]*=factor;
}
__global__ void add(Real* a, Real* b, Real c){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  a[index]+=b[index]*c;
}

__global__ void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  int marginx = (cuda_row-row)/2;
  int marginy = (cuda_column-col)/2;
  if(x<marginx || x >= cuda_row-marginx || y < marginy || y >= cuda_column-marginy){
    objectWave[index].x = objectWave[index].y = 0;
    return;
  }
  int targetindex = (x-marginx)*col + y-marginy;
  Real mod = sqrtf(max(0.,d_intensity[targetindex]));
  Real phase = d_phase? (d_phase[targetindex]-0.5)*2*M_PI : 0;
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
}

__global__ void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  Real marginratio = (1-1./oversampling)/2;
  int marginx = marginratio*cuda_row;
  int marginy = marginratio*cuda_column;
  if(x<marginx || x >= cuda_row-marginx || y < marginy || y >= cuda_column-marginy){
    objectWave[index].x = objectWave[index].y = 0;
    return;
  }
  int targetindex = (x-marginx)*ceil(cuda_column/oversampling) + y-marginy;
  Real mod = sqrtf(max(0.,d_intensity[targetindex]));
  Real phase = d_phase? (d_phase[targetindex]-0.5)*2*M_PI : 0;
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
}

__global__ void initRand(curandStateMRG32k3a *state){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  curand_init(1,index,0,&state[index]);
}

__global__ void applyPoissonNoise_WO(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(scale==0) scale = cuda_scale;
  wave[index]+=scale*(curand_poisson(&state[index], noiseLevel)-noiseLevel)/cuda_rcolor;
}

__global__ void applyPoissonNoise(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  curand_init(1,index,0,&state[index]);
  if(scale==0) scale = cuda_scale;
  wave[index]+=scale*(curand_poisson(&state[index], noiseLevel)-noiseLevel)/cuda_rcolor;
}

__global__  void getMod(Real* mod, complexFormat* amp){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  mod[index] = cuCabsf(amp[index]);
}
__global__  void getMod2(Real* mod2, complexFormat* amp){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  mod2[index] = pow(amp[index].x,2)+pow(amp[index].y,2);
}

__global__ void applyMod(complexFormat* source, Real* target, Real *bs, bool loose, int iter, int noiseLevel){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  Real maximum = cuda_scale*0.95;
  int index = x*cuda_column + y;
  Real mod2 = target[index];
  if(mod2<0) mod2=0;
  if(loose && bs && bs[index]>0.5) {
    //if(iter > 500) return;
    //else mod2 = maximum+1;
    return;
  }
  Real tolerance = 1./cuda_rcolor*cuda_scale+1.5*sqrtf(noiseLevel)/cuda_rcolor; // fluctuation caused by bit depth and noise
  complexFormat sourcedata = source[index];
  Real ratiox = 1;
  Real ratioy = 1;
  Real srcmod2 = sourcedata.x*sourcedata.x + sourcedata.y*sourcedata.y;
  if(mod2>=maximum) {
    if(loose) mod2 = max(maximum,srcmod2);
    else tolerance*=1000;
  }
  Real diff = mod2-srcmod2;
  if(diff>tolerance){
    ratioy=ratiox = sqrt((mod2-tolerance)/srcmod2);
  }else if(diff < -tolerance ){
    ratioy=ratiox = sqrt((mod2+tolerance)/srcmod2);
  }
  if(srcmod2 == 0){
    ratiox = sqrt(mod2);
    ratioy = 0;
  }
  source[index].x = ratiox*sourcedata.x;
  source[index].y = ratioy*sourcedata.y;
}
__global__ void add(complexFormat* a, complexFormat* b, Real c ){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  a[index].x+=b[index].x*c;
  a[index].y+=b[index].y*c;
}
__global__ void applyRandomPhase(complexFormat* wave, Real* beamstop, curandStateMRG32k3a *state){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  complexFormat tmp = wave[index];
  if(beamstop && beamstop[index]>cuda_threshold) {
    tmp.x = tmp.y = 0;
  }
  else{
    Real mod = cuCabsf(wave[index]);
    curand_init(1,index,0,&state[index]);
    //curand_poisson(&state[index], noiseLevel) can do poission noise
    Real randphase = curand_uniform(&state[index]);
    tmp.x = mod*cos(randphase);
    tmp.y = mod*sin(randphase);
  }
  wave[index] = tmp;
}
