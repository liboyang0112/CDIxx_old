#include <complex>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <random>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <libconfig.h++>
#include "cufft.h"
#include "common.h"
#include <ctime>
#include "cudaConfig.h"
#include "readConfig.h"
#include "tvFilter.h"
#include "cuPlotter.h"

#include <cub/device/device_reduce.cuh>
#include <curand_kernel.h>
struct CustomMax
{
  __device__ __forceinline__
    Real operator()(const Real &a, const Real &b) const {
      return (b > a) ? b : a;
    }
};

Real findMax(Real* d_in, int num_items)
{
  Real *d_out = NULL;
  cudaMalloc((void**)&d_out, sizeof(Real));

  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;
  CustomMax max_op;
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, 0));
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, 0));
  Real output;
  cudaMemcpy(&output, d_out, sizeof(Real), cudaMemcpyDeviceToHost);

  if (d_out) cudaFree(d_out);
  if (d_temp_storage) cudaFree(d_temp_storage);
  return output;
}

//#define Bits 16

Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*pi*sigma*sigma)*gaussian(x,y,sigma);
}

class rect{
  public:
    int startx;
    int starty;
    int endx;
    int endy;
    __device__ __host__ bool isInside(int x, int y){
      if(x > startx && x <= endx && y > starty && y <= endy) return true;
      return false;
    }
};
class C_circle{
  public:
    int x0;
    int y0;
    Real r;
    __device__ __host__ bool isInside(int x, int y){
      Real dr = sqrt(pow(x-x0,2)+pow(y-y0,2));
      if(dr < r) return true;
      return false;
    }
};

template <typename sptType>
__global__ void createMask(Real* data, sptType* spt, bool isFrequency=0){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
  }
  data[index]=spt->isInside(x,y);
}

template <typename T>
__global__ void cudaConvertFO(T* data){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= (cuda_row>>1) || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int indexp = (x+(cuda_row>>1))*cuda_column + (y >= (cuda_column>>1)? y-(cuda_column>>1): (y+(cuda_column>>1)));
  T tmp = data[index];
  data[index]=data[indexp];
  data[indexp]=tmp;
}

__global__ void applyNormConvertFO(complexFormat* data, Real factor){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= (cuda_row>>1) || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int indexp = (x+(cuda_row>>1))*cuda_column + (y >= (cuda_column>>1)? y-(cuda_column>>1): (y+(cuda_column>>1)));
  complexFormat tmp = data[index];
  data[index].x=data[indexp].x*factor;
  data[index].y=data[indexp].y*factor;
  data[indexp].x=tmp.x*factor;
  data[indexp].y=tmp.y*factor;
}

__global__  void initESW(complexFormat* ESW, Real* mod, complexFormat* amp){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  auto tmp = amp[index];
  if(cuCabsf(tmp)<=1e-10) {
    ESW[index] = tmp;
    return;
  }
  Real factor = sqrtf(mod[index])/cuCabsf(tmp)-1;
  ESW[index].x = factor*tmp.x;
  ESW[index].y = factor*tmp.y;
}
__global__  void applyESWMod(complexFormat* ESW, Real* mod, complexFormat* amp, int noiseLevel){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  Real tolerance = 0;//1./cuda_rcolor*cuda_scale+1.5*sqrtf(noiseLevel)/cuda_rcolor; // fluctuation caused by bit depth and noise
  int index = x*cuda_column + y;
  auto tmp = amp[index];
  auto sum = cuCaddf(ESW[index],tmp);
  Real factor = 0;
  if(fabs(cuCabsf(sum))>1e-10){
    //factor = mod[index]/cuCabsf(sum);
    Real mod2 = mod[index];
    Real mod2s = sum.x*sum.x+sum.y*sum.y;
    if(mod2+tolerance < mod2s) factor = sqrt((mod2+tolerance)/mod2s);
    else if(mod2-tolerance > mod2s) factor = sqrt((mod2-tolerance)/mod2s);
    else factor=1;
  }
  //if(mod[index] >= 0.99) factor = max(0.99/cuCabsf(sum), 1.);
  //printf("factor=%f, mod=%f, sum=%f\n", factor, mod[index], cuCabsf(sum));
  ESW[index].x = factor*sum.x-tmp.x;
  ESW[index].y = factor*sum.y-tmp.y;
}

__global__ void applyPoissonNoise(Real* wave, Real noiseLevel, curandStateMRG32k3a *state){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  curand_init(1,index,0,&state[index]);
  wave[index]+=(curand_poisson(&state[index], noiseLevel)-noiseLevel)/cuda_rcolor;
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

__global__  void applyESWSupport(complexFormat* ESW, complexFormat* ISW, complexFormat* ESWP, Real* length){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  auto tmp = ISW[index];
  auto tmp2 = ESWP[index];
  auto sum = cuCaddf(tmp,ESWP[index]);
  //these are for amplitude modulation only
  Real prod = tmp.x*tmp2.x+tmp.y*tmp2.y;
  if(prod>0) prod=0;
  if(prod<-2) prod = -2;
  auto rmod2 = 1./(tmp.x*tmp.x+tmp.y*tmp.y);
  ESW[index].x = prod*tmp.x*rmod2;
  ESW[index].y = prod*tmp.y*rmod2;
  /*
     if(cuCabsf(tmp) > cuCabsf(sum)) {
     ESW[index] = ESWP[index];
     length[index] = 0;
     return;
     }
     Real factor = cuCabsf(tmp)/cuCabsf(sum);
     if(x<cuda_row/3||x>cuda_row*2/3||y<cuda_column||y>2*cuda_column/3) factor = 0;
     ESW[index].x = factor*sum.x-tmp.x;
     ESW[index].y = factor*sum.y-tmp.y;

     ESW[index].x -= cuda_beta_HIO*(1-factor)*sum.x;
     ESW[index].y -= cuda_beta_HIO*(1-factor)*sum.y;
     length[index] = 1;
   */
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
__global__ void add(complexFormat* a, complexFormat* b){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  a[index]=cuCaddf(a[index],b[index]);
}

__global__ void takeMod2Diff(complexFormat* a, Real* b, Real *output, Real *bs){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  output[index] = ((bs&&bs[index]>cuda_threshold)|| b[index]>0.95)?0:(b[index]-pow(a[index].x,2)-pow(a[index].y,2));
}

__global__ void takeMod2Sum(complexFormat* a, Real* b){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  Real tmp = b[index]+pow(a[index].x,2)+pow(a[index].y,2);
  if(tmp<0) tmp=0;
  b[index] = tmp;
}

__global__ void calcESW(complexFormat* sample, complexFormat* ISW){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  complexFormat tmp = sample[index];
  Real ttmp = tmp.y;
  tmp.y=1-tmp.x;   // We are ignoring the factor (-i) each time we do fresnel propagation, which causes this transform in the ISW. ISW=iA ->  ESW=(O-1)A=(i-iO)ISW
                   // When we do reconstruction, ESW is reconstructed, so O = 1+i*ESW/ISW
  tmp.x=ttmp;
  sample[index]=cuCmulf(tmp,ISW[index]);
}

__global__ void calcO(complexFormat* ESW, complexFormat* ISW){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(cuCabsf(ISW[index])<0.1) {
    ESW[index].x = 1;
    ESW[index].y = 0;
    return;
  }
  complexFormat tmp = cuCdivf(ESW[index],ISW[index]);
  /*
     Real ttmp = tmp.y;
     tmp.y=tmp.x;   
     tmp.x=1-ttmp;
   */
  ESW[index].x=tmp.x+1;
  ESW[index].y=tmp.y;
}

__global__ void applyMod(complexFormat* source, Real* target, Real *bs = 0, bool loose=0, int iter = 0, int noiseLevel = 0){
  assert(source!=0);
  assert(target!=0);
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  Real maximum = pow(mergeDepth,2)*cuda_scale*0.95;
  int index = x*cuda_column + y;
  Real mod2 = target[index];
  if(mod2<0) mod2=0;
  if(loose && bs && bs[index]>cuda_threshold) {
    if(iter > 500) return;
    else mod2 = maximum+1;
  }
  Real tolerance = 0;//1./cuda_rcolor*cuda_scale+1.5*sqrtf(noiseLevel)/cuda_rcolor; // fluctuation caused by bit depth and noise
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


__global__ void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  Real marginratio = (1-1./oversampling)/2;
  Real marginx = marginratio*cuda_row;
  Real marginy = marginratio*cuda_column;
  if(x<marginx || x >= cuda_row-marginx || y < marginy || y >= cuda_column-marginy){
    objectWave[index].x = objectWave[index].y = 0;
    return;
  }
  int targetindex = (x-marginx)*cuda_column/oversampling + y-marginy;
  Real mod = sqrtf(max(0.,d_intensity[targetindex]));
  Real phase = d_phase? d_phase[targetindex] : 0;
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
}

__global__ void multiplyPatternPhase_Device(complexFormat* amp, Real r_d_lambda, Real d_r_lambda){ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  Real phase = (pow(x-(cuda_row>>1),2)+pow(y-(cuda_column>>1),2))*r_d_lambda+d_r_lambda;
  int index = x*cuda_column + y;
  complexFormat p = {cos(phase),sin(phase)};
  amp[index] = cuCmulf(amp[index], p);
}

__global__ void multiplyFresnelPhase_Device(complexFormat* amp, Real phaseFactor){ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  Real phase = phaseFactor*(pow(x-(cuda_row>>1),2)+pow(y-(cuda_column>>1),2));
  complexFormat p = {cos(phase),sin(phase)};
  amp[index] = cuCmulf(amp[index], p);
}
__device__ void ApplyHIOSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
    rhonp1.y -= beta*rhoprime.y;
  }
}

__device__ void ApplyRAARSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = beta*rhonp1.x+(1-2*beta)*rhoprime.x;
    rhonp1.y = beta*rhonp1.y+(1-2*beta)*rhoprime.y;
  }
}

__device__ void ApplyERSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
}


__device__ void ApplyPOSHIOSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime, Real beta){
  if(rhoprime.x > 0 && (insideS/* || rhoprime[0]<30./rcolor*/)){
    rhonp1.x = rhoprime.x;
  }else{
    rhonp1.x -= beta*rhoprime.x;
  }
  rhonp1.y -= beta*rhoprime.y;
}


class experimentConfig : public readConfig{
  public:
    //calculated later by init function, image size dependant
    experimentConfig(const char* configfile):readConfig(configfile){}

    cuPlotter plt;

    Real enhancement = 0;
    Real forwardFactor = 0;
    Real fresnelFactor = 0;
    Real inverseFactor = 0;

    Real enhancementKCDI = 0;
    Real forwardFactorKCDI = 0;
    Real fresnelFactorKCDI = 0;
    Real inverseFactorKCDI = 0;

    Real   enhancementMid = 0;
    Real forwardFactorMid = 0;
    Real fresnelFactorMid = 0;
    Real inverseFactorMid = 0;
    Real resolution = 0;
    int row = 0;
    int column = 0;

    Real dKCDI = 0;

    Real* patternData = 0;
    complexFormat* patternWave = 0;
    complexFormat* objectWave = 0;
    Real* KCDIpatternData = 0;
    complexFormat* KCDIpatternWave = 0;
    complexFormat* KCDIobjectWave = 0;

    Real* support = 0;
    Real* beamstop = 0;


    template<typename sptType>
    void createBeamStop(sptType *spt){
      sptType *cuda_spt;
      gpuErrchk(cudaMalloc((void**)&cuda_spt,sizeof(sptType)));
      cudaMemcpy(cuda_spt, spt, sizeof(sptType), cudaMemcpyHostToDevice);
      createMask<<<numBlocks,threadsPerBlock>>>(beamstop, cuda_spt,1);
      cudaFree(cuda_spt);
    }

    void propagate(complexFormat* datain, complexFormat* dataout, bool isforward){
      myCufftExec( *plan, datain, dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
      applyNorm<<<numBlocks,threadsPerBlock>>>(dataout, isforward? forwardFactor: inverseFactor);
    }
    void propagateKCDI(complexFormat* datain, complexFormat* dataout, bool isforward){
      myCufftExec( *plan, datain, dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
      applyNorm<<<numBlocks,threadsPerBlock>>>(dataout, isforward? forwardFactorKCDI: inverseFactorKCDI);
    }
    void propagateMid(complexFormat* datain, complexFormat* dataout, bool isforward){
      myCufftExec( *plan, datain, dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
      applyNorm<<<numBlocks,threadsPerBlock>>>(dataout, isforward? forwardFactorMid: inverseFactorMid);
    }
    void multiplyPatternPhase(complexFormat* amp, Real distance){
      multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>(amp, pixelsize*pixelsize*M_PI/(distance*lambda), 2*distance*M_PI/lambda);
    }
    void multiplyPatternPhaseMid(complexFormat* amp, Real distance){
      multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>(amp, resolution*resolution*M_PI/(distance*lambda), 2*distance*M_PI/lambda);
    }
    void multiplyFresnelPhase(complexFormat* amp, Real distance){
      Real fresfactor = M_PI*lambda*distance/(pow(pixelsize*row,2));
      multiplyFresnelPhase_Device<<<numBlocks,threadsPerBlock>>>(amp, fresfactor);
    }
    void multiplyFresnelPhaseMid(complexFormat* amp, Real distance){
      Real fresfactor = M_PI*lambda*distance/(pow(resolution*row,2));
      multiplyFresnelPhase_Device<<<numBlocks,threadsPerBlock>>>(amp, fresfactor);
    }
    void allocateMem(){
      cudaMalloc((void**)&support, row*column*sizeof(Real)); //preallocation of support and beamstop
      cudaMalloc((void**)&beamstop, row*column*sizeof(Real));
      cudaMalloc((void**)&objectWave, row*column*sizeof(complexFormat));
      cudaMalloc((void**)&patternWave, row*column*sizeof(complexFormat));
      cudaMalloc((void**)&patternData, row*column*sizeof(Real));
      init_cuda_image(row,column,rcolor);
      plt.init(row,column);
    }
    void readObjectWave(){
      Real* intensity = readImage(common.Intensity.c_str(), row, column);
      size_t sz = row*column*sizeof(Real);
      row*=oversampling;
      column*=oversampling;
      allocateMem();
      Real* d_intensity = support; //use the memory allocated;
      cudaMemcpy(d_intensity, intensity, sz, cudaMemcpyHostToDevice);
      Real* d_phase = 0;
      if(phaseModulation) {
        Real* phase = readImage(common.Intensity.c_str(), row, column);
        d_phase = beamstop;
        cudaMemcpy(d_phase, phase, sz, cudaMemcpyHostToDevice);
      }
      createWaveFront<<<numBlocks,threadsPerBlock>>>(d_intensity, d_phase, objectWave, oversampling);
      if(isFresnel) multiplyFresnelPhase(objectWave, d);
       // if(setups.useRectHERALDO){
       //   pixeltype *rowp;
       //   for(int i = 0; i < row ; i++){
       //     rowp = intensity.ptr<pixeltype>(i);
       //     for(int j = 0; j < column ; j++){
       //       if(i > row/2 || j > column/2) rowp[j] = rcolor-1;
       //     }
       //   }
       // }
      //if(setups.useGaussionLumination){
      //  //setups.spt = &re;
      //  //if(!setups.useShrinkMap) setups.spt = &cir3;
      //  //diffraction image, either from simulation or from experiments.
      //  auto f = [&](int x, int y, fftw_format &data){
      //    auto tmp = (complex<Real>*)&data;
      //    bool inside = cir3.isInside(x,y);
      //    if(!inside) *tmp = 0.;
      //    *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir3.r);
      //    //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
      //  };
      //  imageLoop<decltype(f)>(gkp1,&f,0);
      //}
      //if(setups.useGaussionHERALDO){
      //  auto f = [&](int x, int y, fftw_format &data){
      //    auto tmp = (complex<Real>*)&data;
      //    if(cir2.isInside(x,y)) 
      //      *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir2.r*4);
      //    else *tmp = gaussian(x-cir2.x0,y-cir2.y0,cir2.r*4);
      //    if(x < row*1/3 && y < row*1/3) *tmp = 0;
      //    //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
      //  };
      //  imageLoop<decltype(f)>(gkp1,&f,0);
      //}
      plt.plotComplex(objectWave, MOD2, 0, 1, "inputIntensity", 0);
      plt.plotComplex(objectWave, PHASE, 0, 1, "inputPhase", 0);
      //if(useRectHERALDO)
      //  setRectHERALDO<<<numBlocks,threadsPerBlock>>>(objectWave, oversampling);
      //}
    }
    void readPattern(){
      Real* pattern = readImage(common.Pattern.c_str(), row, column);
      allocateMem();
      cudaMemcpy(patternData, pattern, row*column*sizeof(Real), cudaMemcpyHostToDevice);
      cudaConvertFO<<<numBlocks,threadsPerBlock>>>(patternData);
    }
    void init(){
      if(runSim) {
        readObjectWave();
      }else{
        readPattern();
      }
      C_circle cir;
      cir.x0=row/2;
      cir.y0=column/2;
      cir.r=beamStopSize;
      createBeamStop(&cir);
      enhancement = pow(pixelsize,2)*sqrt(row*column)/(lambda*d); // this guarentee energy conservation
      enhancement *= exposure; 
      fresnelFactor = lambda*d/pow(pixelsize,2)/row/column;
      forwardFactor = fresnelFactor*enhancement;
      inverseFactor = 1./row/column/forwardFactor;
      if(doKCDI) {
        Real k = row*pow(pixelsize,2)/(lambda*d);
        dKCDI = d*k/(k+1);
        resolution = lambda*dKCDI/(row*pixelsize);
        printf("Resolution=%4.2fum\n", resolution);
        enhancementKCDI = pow(pixelsize,2)*sqrt(row*column)/(lambda*dKCDI); // this guarentee energy conservation
        enhancementKCDI *= exposureKCDI;
        fresnelFactorKCDI = lambda*dKCDI/pow(pixelsize,2)/row/column;
        forwardFactorKCDI = fresnelFactorKCDI*enhancementKCDI;
        inverseFactorKCDI = 1./row/column/forwardFactorKCDI;

        enhancementMid = pow(resolution,2)*sqrt(row*column)/(lambda*(d-dKCDI)); // this guarentee energy conservation
        fresnelFactorMid = lambda*(d-dKCDI)/pow(resolution,2)/row/column;
        forwardFactorMid = fresnelFactorMid*enhancementMid;
        inverseFactorMid = 1./row/column/forwardFactorMid;
      }
      curandStateMRG32k3a *devstates;
      cudaMalloc((void **)&devstates, column * row * sizeof(curandStateMRG32k3a));
      if(runSim){
        propagate(objectWave, patternWave, 1);
        getMod2<<<numBlocks,threadsPerBlock>>>(patternData, patternWave);
        if(simCCDbit){
          applyPoissonNoise<<<numBlocks,threadsPerBlock>>>(patternData, noiseLevel, devstates);
        }
      }
      if(restart){
        complexFormat *wf = (complexFormat*) readComplexImage(common.restart.c_str());
        cudaMemcpy(patternWave, wf, row*column*sizeof(complexFormat), cudaMemcpyHostToDevice);
        free(wf);
      }else {
        if(!runSim) createWaveFront<<<numBlocks,threadsPerBlock>>>(patternData, 0, patternWave, 1);
        applyRandomPhase<<<numBlocks,threadsPerBlock>>>(patternWave, useBS?beamstop:0, devstates);
        plt.plotComplex(patternWave, MOD2, 1, 1, "init_logpattern", 1);
        plt.plotComplex(patternWave, MOD2, 1, 1, "init_pattern", 0);
      }
      cudaFree(devstates);
      rect re;
      re.startx = (oversampling-1)/2*row/oversampling-1;
      re.starty = (oversampling-1)/2*column/oversampling-1;
      re.endx = row-re.startx;
      re.endy = column-re.starty;
      rect *cuda_spt;
      gpuErrchk(cudaMalloc((void**)&cuda_spt,sizeof(rect)));
      cudaMemcpy(cuda_spt, &re, sizeof(rect), cudaMemcpyHostToDevice);
      createMask<<<numBlocks,threadsPerBlock>>>(support, cuda_spt,0);
      cudaFree(cuda_spt);
    }
};

__global__ void applySupport(complexFormat *gkp1, complexFormat *gkprime, Real* objMod, Real *spt, int iter = 0, Real fresnelFactor = 0){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = x*cuda_column + y;
  if(x >= cuda_row || y >= cuda_column) return;

  bool inside = spt[index] > cuda_threshold;
  complexFormat &gkp1data = gkp1[index];
  complexFormat &gkprimedata = gkprime[index];
  if(iter < 300) ApplyRAARSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  else ApplyERSupport(inside,gkp1data,gkprimedata);
  if(fresnelFactor*(cuda_row*cuda_row+cuda_column*cuda_column)>1 && iter < 100) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(pow(x-(cuda_row>>1),2)+pow(y-(cuda_column>>1),2));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
  objMod[index] = cuCabsf(gkp1data);
}
complexFormat* phaseRetrieve(experimentConfig &setups){
  int row = setups.row;
  int column = setups.column;
  bool useShrinkMap = setups.useShrinkMap;
  bool useBS = setups.useBS;
  Real *cir = setups.beamstop;
  Real beta = -1;
  Real gammas = -1./beta;
  Real gammam = 1./beta;
  Real gaussianSigma = 2;

  size_t sz = row*column*sizeof(complexFormat);
  complexFormat *cuda_gkprime;
  complexFormat *cuda_gkp1 = setups.objectWave;
  complexFormat *cuda_fftresult = setups.patternWave;
  Real *cuda_targetfft = setups.patternData;

  Real *cuda_diff;
  Real *cuda_objMod;
  cudaMalloc((void**)&cuda_diff, sz/2);
  cudaMalloc((void**)&cuda_gkprime, sz);
  cudaMalloc((void**)&cuda_objMod, sz/2);
  cudaMemcpy(cuda_diff, cuda_targetfft, sz/2, cudaMemcpyDeviceToDevice);

  inittvFilter(row,column);
  for(int iter = 0; iter < setups.nIter; iter++){
    //start iteration
    if(iter%100==0) {
      std::string iterstr = to_string(iter);
      if(setups.saveIter){
        setups.plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+iterstr).c_str(), 0);
        setups.plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+iterstr).c_str(), 0);
        setups.plt.plotComplex(cuda_fftresult, MOD2, 1, 1, ("recon_pattern"+iterstr).c_str(), 1);
      }
      if(iter>0){
        takeMod2Diff<<<numBlocks,threadsPerBlock>>>(cuda_fftresult,cuda_targetfft, cuda_diff, useBS? cir:0);
        cudaConvertFO<<<numBlocks,threadsPerBlock>>>(cuda_diff);
        tvFilterWrap(cuda_diff, setups.noiseLevel, 100);
        setups.plt.plotFloat(cuda_diff, MOD, 0, 1, ("smoothed"+iterstr).c_str(), 1);
        cudaConvertFO<<<numBlocks,threadsPerBlock>>>(cuda_diff);
        takeMod2Sum<<<numBlocks,threadsPerBlock>>>(cuda_fftresult, cuda_diff);
      }
    }
    applyMod<<<numBlocks,threadsPerBlock>>>(cuda_fftresult,cuda_diff, useBS? cir:0, !setups.reconAC || iter > 1000,iter, setups.noiseLevel);
    setups.propagate(cuda_fftresult, cuda_gkprime, 0);
    applySupport<<<numBlocks,threadsPerBlock>>>(cuda_gkp1, cuda_gkprime, cuda_objMod,setups.support,iter,setups.isFresnel? setups.fresnelFactor:0);
    setups.propagate( cuda_gkp1, cuda_fftresult, 1);
    //update mask
    if((iter > 0 && iter < 200) && iter%20==0 && useShrinkMap){
      int size = floor(gaussianSigma*6); // r=3 sigma to ensure the contribution outside kernel is negligible (0.01 of the maximum)
      size = size/2;
      int width = size*2+1;
      Real* gaussianKernel;
      Real* d_gaussianKernel;
      int kernelsz = width*width*sizeof(Real);
      gaussianKernel = (Real*) malloc(kernelsz);
      cudaMalloc((void**)&d_gaussianKernel, kernelsz);
      Real total = 0;
      Real weight;
      for(int i = 0; i < width*width; i++) {
	      weight = gaussian((i/width-size),i%width-size, gaussianSigma);
	      total+= weight;
	      gaussianKernel[i] = weight;
      }
      for(int i = 0; i < width*width; i++)
	      gaussianKernel[i] /= total;
      cudaMemcpy(d_gaussianKernel, gaussianKernel, kernelsz, cudaMemcpyHostToDevice);
      free(gaussianKernel);
      applyConvolution<<<numBlocks,threadsPerBlock, pow(size*2+threadsPerBlock.x,2)*sizeof(Real)>>>(cuda_objMod, setups.support, d_gaussianKernel, size, size);
      cudaFree(d_gaussianKernel);

      Real threshold = findMax(setups.support, row*column)*setups.shrinkThreshold;
      cudaMemcpyToSymbol(cuda_threshold, &threshold, sizeof(threshold));

      if(gaussianSigma>1.5) {
        gaussianSigma*=0.99;
      }
    }
  }
  setups.plt.plotComplex(cuda_fftresult, MOD2, 1, 1, "recon_pattern", 1);
  applyMod<<<numBlocks,threadsPerBlock>>>(cuda_fftresult,cuda_targetfft,useBS?cir:0,1,setups.nIter, setups.noiseLevel);
  setups.plt.plotComplex(cuda_gkp1, MOD2, 0, 1, "recon_intensity", 0);
  setups.plt.plotComplex(cuda_gkp1, PHASE, 0, 1, "recon_phase", 0);
  cudaFree(cuda_gkp1);
  cudaFree(cuda_targetfft);
  cudaFree(cuda_gkprime);
  cudaFree(cuda_objMod);

  return cuda_fftresult;
}


__global__ void applyAutoCorrelationMod(complexFormat* source,complexFormat* target, Real *bs = 0){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  Real targetdata = target[index].x;
  Real retval = targetdata;
  source[index].y = 0;
  Real maximum = pow(mergeDepth,2)*cuda_scale*0.99;
  Real sourcedata = source[index].x;
  Real tolerance = 0.5/cuda_rcolor*cuda_scale;
  Real diff = sourcedata-targetdata;
  if(bs && bs[index]>cuda_threshold) {
    if(targetdata<0) target[index].x = 0;
    return;
  }
  if(diff>tolerance){
    retval = targetdata+tolerance;
  }else if(diff < -tolerance ){
    retval = targetdata-tolerance;
  }else{
    retval = targetdata;
  }
  if(targetdata>=maximum) {
    retval = max(sourcedata,maximum);
  }
  source[index].x = retval;
}

template <typename sptType>
__global__ void applyERACSupport(complexFormat* data,complexFormat* prime,sptType *spt, Real* objMod){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(!spt->isInside(x,y)){
    data[index].x = 0;
    data[index].y = 0;
  }
  else{
    data[index].x = prime[index].x;
    data[index].y = prime[index].y;
  }
  objMod[index] = cuCabsf(data[index]);
}

template <typename sptType>
__global__ void applyHIOACSupport(complexFormat* data,complexFormat* prime, sptType *spt, Real *objMod){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(!spt->isInside(x,y)){
    data[index].x -= prime[index].x;
  }
  else{
    data[index].x = prime[index].x;
  }
  data[index].y -= prime[index].y;
  objMod[index] = cuCabsf(data[index]);
}

int main(int argc, char** argv )
{
  experimentConfig setups(argv[1]);
  if(argc < 2){
    printf("please feed the object intensity and phase image\n");
  }
  if(setups.runSim) setups.d = setups.oversampling*setups.pixelsize*setups.beamspotsize/setups.lambda; //distance to guarentee setups.oversampling
  setups.init();

  //-----------------------configure experiment setups-----------------------------
  printf("Imaging distance = %4.2fcm\n", setups.d*1e-4);
  printf("forward norm = %f\n", setups.forwardFactor);
  printf("backward norm = %f\n", setups.inverseFactor);
  printf("fresnel factor = %f\n", setups.fresnelFactor);
  printf("enhancement = %f\n", setups.enhancement);

  printf("KCDI Imaging distance = %4.2fcm\n", setups.dKCDI*1e-4);
  printf("KCDI forward norm = %f\n", setups.forwardFactorKCDI);
  printf("KCDI backward norm = %f\n", setups.inverseFactorKCDI);
  printf("KCDI fresnel factor = %f\n", setups.fresnelFactorKCDI);
  printf("KCDI enhancement = %f\n", setups.enhancementKCDI);

  Real fresnelNumber = pi*pow(setups.beamspotsize,2)/(setups.d*setups.lambda);
  printf("Fresnel Number = %f\n",fresnelNumber);
  Real beta_HIO = 0.9;
  cudaMemcpyToSymbol(cuda_beta_HIO,&beta_HIO,sizeof(beta_HIO));
  int sz = setups.row*setups.column*sizeof(complexFormat);
  if(setups.doIteration) {
    phaseRetrieve(setups); 
    void* outputData = malloc(sz);
    cudaMemcpy(outputData, setups.patternWave, sz, cudaMemcpyDeviceToHost);
    writeComplexImage(setups.common.restart.c_str(), outputData, setups.row, setups.column);//save the step
  }

  //Now let's do KCDI
  if(setups.doKCDI){ 
    complexFormat* cuda_KCDIAmp, *cuda_ESW, *cuda_debug, *cuda_ESWP, *cuda_ESWPattern, *cuda_KCDIAmp_SIM;
    Real* cuda_KCDImod;
    gpuErrchk(cudaMalloc((void**)&cuda_KCDImod, sz/2));
    gpuErrchk(cudaMalloc((void**)&cuda_KCDIAmp_SIM, sz));
    gpuErrchk(cudaMalloc((void**)&cuda_ESW, sz));
    gpuErrchk(cudaMalloc((void**)&cuda_ESWP, sz));
    gpuErrchk(cudaMalloc((void**)&cuda_ESWPattern, sz));
    gpuErrchk(cudaMalloc((void**)&cuda_debug, sz));
    gpuErrchk(cudaMalloc((void**)&cuda_KCDIAmp, sz));
    if(setups.runSim){
      cudaMemcpy(cuda_KCDIAmp, setups.objectWave, sz, cudaMemcpyDeviceToDevice);

      setups.multiplyFresnelPhaseMid(cuda_KCDIAmp, setups.d-setups.dKCDI);
      setups.propagateMid(cuda_KCDIAmp, cuda_KCDIAmp, 1);
      cudaConvertFO<<<numBlocks,threadsPerBlock>>>(cuda_KCDIAmp);
      setups.multiplyPatternPhaseMid(cuda_KCDIAmp, setups.d-setups.dKCDI);

      setups.plt.plotComplex(cuda_KCDIAmp, MOD2, 0, 1, "ISW");
      int row, column;
      Real* KCDIInput = readImage(setups.KCDI.Intensity.c_str(), row, column);
      
      cudaMemcpy(cuda_KCDImod, KCDIInput, sz/2, cudaMemcpyHostToDevice);
      free(KCDIInput);
      createWaveFront<<<numBlocks,threadsPerBlock>>>(cuda_KCDImod, 0, cuda_ESW, setups.oversampling);
      setups.plt.plotComplex(cuda_ESW, MOD2, 0, 1, "KCDIsample", 0);
      calcESW<<<numBlocks,threadsPerBlock>>>(cuda_ESW, cuda_KCDIAmp);
      setups.plt.plotComplex(cuda_ESW, MOD2, 0, 1, "ESW");

      setups.multiplyFresnelPhase(cuda_ESW, setups.dKCDI);
      setups.propagateKCDI(cuda_ESW, cuda_ESW, 1);
      //setups.multiplyPatternPhase(cuda_ESW, setups.dKCDI); //the same effect as setups.multiplyPatternPhase(cuda_KCDIAmp, -setups.dKCDI);
      setups.plt.plotComplex(cuda_ESW, MOD2, 0, 1, "ESWPattern",1);

      cudaMemcpy(cuda_KCDIAmp, setups.objectWave, sz, cudaMemcpyDeviceToDevice);
      setups.multiplyFresnelPhase(cuda_KCDIAmp, setups.d);
      setups.propagateKCDI(cuda_KCDIAmp, cuda_KCDIAmp, 1); // equivalent to fftresult
      cudaMemcpy(cuda_KCDIAmp_SIM,cuda_KCDIAmp, sz, cudaMemcpyDeviceToDevice);
      cudaConvertFO<<<numBlocks,threadsPerBlock>>>(cuda_KCDIAmp);
      setups.multiplyPatternPhase(cuda_KCDIAmp, setups.d);
      setups.multiplyPatternPhase(cuda_KCDIAmp, -setups.dKCDI);

      setups.plt.plotComplex(cuda_KCDIAmp, MOD2, 0, 1, "srcPattern",1);

      add<<<numBlocks,threadsPerBlock>>>(cuda_KCDIAmp, cuda_ESW);
      getMod2<<<numBlocks,threadsPerBlock>>>(cuda_KCDImod, cuda_KCDIAmp);
      curandStateMRG32k3a *devstates;
      cudaMalloc((void **)&devstates, column * row * sizeof(curandStateMRG32k3a));
      applyPoissonNoise<<<numBlocks,threadsPerBlock>>>(cuda_KCDImod, setups.noiseLevel, devstates);
      cudaFree(devstates);
      setups.plt.plotComplex(cuda_KCDImod, MOD, 0, 1, "KCDI_logintensity", 1);
      setups.plt.plotComplex(cuda_KCDIAmp, PHASE, 0, 1, "KCDI_phase", 0);
      setups.plt.plotComplex(cuda_KCDImod, MOD, 0, 1, setups.KCDI.Pattern.c_str(), 0);
    }else{
      int row, column;
      Real* pattern = readImage(setups.KCDI.Pattern.c_str(),row,column); //reconstruction is better after integerization
      cudaMemcpy(cuda_KCDImod, pattern, sz, cudaMemcpyHostToDevice);
      free(pattern);
    }
    //KCDI reconstruction needs:
    //  1. fftresult from previous phaseRetrieve
    //  2. KCDI pattern.
    //Storage:
    //  1. Amp_i
    //  2. ESW
    //  3. ISW
    //  4. sqrt(KCDImod2)
    //
    //KCDI reconstruction procedure:
    //      Amp_i = PatternPhase_d/PatternPhase_KCDI*fftresult
    //  1. ISW = IFFT(Amp_i)
    //  2. ESW = IFFT{(sqrt(KCDImod2)/mod(Amp_i)-1)*(Amp_i)}
    //  3. validate: |FFT(ISW+ESW)| = sqrt(KCDImod2)
    //  4. if(|ESW+ISW| > |ISW|) ESW' = |ISW|/|ESW+ISW|*(ESW+ISW)-ISW
    //     else ESW'=ESW
    //     ESW'->ESW
    //  5. ESWfft = FFT(ESW)
    //  6. ESWfft' = sqrt(KCDImod2)/|Amp_i+ESWfft|*(Amp_i+ESWfft)-Amp_i
    //  7. ESW = IFFT(ESWfft')
    //  repeat from step 4

    complexFormat* cuda_ISW;
    gpuErrchk(cudaMalloc((void**)&cuda_ISW, sz));
    cudaMemcpy(cuda_KCDIAmp, setups.patternWave, sz, cudaMemcpyDeviceToDevice);
    //cudaMemcpy(cuda_KCDIAmp, cuda_KCDIAmp_SIM, sz, cudaMemcpyHostToDevice);
    cudaConvertFO<<<numBlocks,threadsPerBlock>>>(cuda_KCDIAmp);
    applyNorm<<<numBlocks,threadsPerBlock>>>(cuda_KCDIAmp, sqrt(setups.exposureKCDI/setups.exposure));
    setups.multiplyPatternPhase(cuda_KCDIAmp, setups.d);
    setups.multiplyPatternPhase(cuda_KCDIAmp, -setups.dKCDI);
    setups.plt.plotComplex(cuda_KCDIAmp, MOD2, 0, 1, "amp",0);

    setups.propagateKCDI(cuda_KCDIAmp, cuda_ISW, 0);
    setups.plt.plotComplex(cuda_ISW, MOD2, 0, 1, "ISW_debug",1);

    initESW<<<numBlocks,threadsPerBlock>>>(cuda_ESW, cuda_KCDImod, cuda_KCDIAmp);
    setups.propagateKCDI(cuda_ESW, cuda_ESW, 0);
    cudaMemcpy(cuda_ESWP, cuda_ESW, sz, cudaMemcpyDeviceToDevice);
    Real *cuda_steplength, *steplength;//, lengthsum;
    steplength = (Real*)malloc(sz/2);
    cudaMalloc((void**)&cuda_steplength, sz/2);
    for(int iter = 0; iter < setups.nIterKCDI ;iter++){
      applyESWSupport<<<numBlocks,threadsPerBlock>>>(cuda_ESW, cuda_ISW, cuda_ESWP, cuda_steplength);
      cudaMemcpy(steplength, cuda_steplength, sz/2, cudaMemcpyDeviceToHost);
      /*
         lengthsum = 0;
         for(int i = 0; i < row*column; i++) lengthsum+=steplength[i];
         if(iter%500==0) printf("step: %d, steplength=%f\n", iter, lengthsum);
         if(lengthsum<1e-6) break;
       */
      setups.propagateKCDI(cuda_ESW, cuda_ESWPattern, 1);
      applyESWMod<<<numBlocks,threadsPerBlock>>>(cuda_ESWPattern, cuda_KCDImod, cuda_KCDIAmp, 0);//setups.noiseLevel);
      setups.propagateKCDI(cuda_ESWPattern, cuda_ESWP, 0);
    }

    //convert from ESW to object
    setups.propagateKCDI(cuda_ESW, cuda_ESWPattern, 1);
    add<<<numBlocks,threadsPerBlock>>>(cuda_ESWPattern,cuda_KCDIAmp);
    setups.plt.plotComplex(cuda_ESWPattern, MOD2, 0, 1, "ESW_pattern_recon", 1);

    setups.plt.plotComplex(cuda_ESWP, MOD2, 0, 1, "ESW_recon");

    //applyESWSupport<<<numBlocks,threadsPerBlock>>>(cuda_ESW, cuda_ISW, cuda_ESWP,cuda_steplength);
    calcO<<<numBlocks,threadsPerBlock>>>(cuda_ESWP, cuda_ISW);
    setups.plt.plotComplex(cuda_ESWP, MOD2, 0, 1, "object");
  }
  return 0;
}

