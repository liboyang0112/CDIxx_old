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
#include "experimentConfig.h"
#include "tvFilter.h"
#include "cuPlotter.h"

#include <cub/device/device_reduce.cuh>
#define ALPHA 0.2
#define BETA 1
struct CustomMax
{
  __device__ __forceinline__
    complexFormat operator()(const complexFormat &a, const complexFormat &b) const {
      Real mod2a = a.x*a.x+a.y*a.y;
      Real mod2b = b.x*b.x+b.y*b.y;
      return (mod2a > mod2b) ? a : b;
    }
};

Real findMax(complexFormat* d_in, int num_items)
{
  complexFormat *d_out = NULL;
  cudaMalloc((void**)&d_out, sizeof(complexFormat));

  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;
  CustomMax max_op;
  complexFormat tmp;
  tmp.x = tmp.y = 0;
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, tmp));
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, tmp));
  complexFormat output;
  cudaMemcpy(&output, d_out, sizeof(complexFormat), cudaMemcpyDeviceToHost);

  if (d_out) cudaFree(d_out);
  if (d_temp_storage) cudaFree(d_temp_storage);
  return output.x*output.x + output.y*output.y;
}

//#define Bits 16

__device__ __host__ Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*M_PI*sigma*sigma)*gaussian(x,y,sigma);
}
__global__ void applySupport(Real* image, Real* support){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  if(support[index] > cuda_threshold) image[index] = 0;
}


__global__ void multiplyProbe(complexFormat* object, complexFormat* probe, complexFormat* U, int shiftx, int shifty, int objrow, int objcol, complexFormat *window = 0){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  complexFormat tmp =  object[(x+shiftx)*objcol+y+shifty];
  if(window) window[index] = tmp;
  U[index] = cuCmulf(probe[index], tmp);
}

__device__ void ePIE(complexFormat &target, complexFormat source, complexFormat &diff, Real maxi, Real param){
  Real denom = param/(maxi);
  source = cuCmulf(cuConjf(source),diff);
  target.x -= source.x*denom;
  target.y -= source.y*denom;
};

__device__ void rPIE(complexFormat &target, complexFormat source, complexFormat &diff, Real maxi, Real param){
  Real denom = source.x*source.x+source.y*source.y;
//  if(denom < 8e-4*maxi) return;
  denom = 1./((1-param)*denom+param*maxi);
  source = cuCmulf(cuConjf(source),diff);
  target.x -= source.x*denom;
  target.y -= source.y*denom;
};

__global__ void updateObject(complexFormat* object, complexFormat* probe, complexFormat* U, int shiftx, int shifty, int objrow, int objcol, Real mod2maxProbe){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int indexO = (x+shiftx)*objcol+y+shifty;
  rPIE(object[indexO], probe[index], U[index], mod2maxProbe, ALPHA);
}

__global__ void updateObjectAndProbe(complexFormat* object, complexFormat* probe, complexFormat* U, int shiftx, int shifty, int objrow, int objcol, Real mod2maxProbe, Real mod2maxObj){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int indexO = (x+shiftx)*objcol+y+shifty;
  complexFormat objectdat= object[indexO];
  complexFormat diff= U[index];
  rPIE(object[indexO], probe[index], diff, mod2maxProbe, ALPHA);
  rPIE(probe[index], objectdat, diff, mod2maxObj, BETA);
}

__global__ void random(complexFormat* object, curandStateMRG32k3a *state){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  curand_init(1,index,0,&state[index]);
  object[index].x = curand_uniform(&state[index]);
  object[index].y = curand_uniform(&state[index]);
}

__global__ void pupilFunc(complexFormat* object){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int shiftx = x - cuda_row/2;
  int shifty = y - cuda_column/2;
  object[index].x = 3*gaussian(shiftx,shifty,cuda_row/8);
  object[index].y = 0;
}

class ptycho : public experimentConfig{
  public:
    int row_O = 512;  //in ptychography this is different from row (the size of probe).
    int column_O = 512;
    int sz = 0;
    int stepSize = 32;
    int dpupil = 100;
    int doPhaseModulationPupil = 0;
    int scanx = 0;
    int scany = 0;
    Real **patterns; //patterns[i*scany+j] points to the address on device to store pattern;
    complexFormat *esw;
    complexFormat *complexCache;
    curandStateMRG32k3a *devstates = 0;

    ptycho(const char* configfile):experimentConfig(configfile){}
    void allocateMem(){
      if(devstates) return;
      gpuErrchk(cudaMalloc((void **)&devstates, column_O * row_O * sizeof(curandStateMRG32k3a)));
      printf("allocating memory\n");
      scanx = (row_O-row)/stepSize+1;
      scany = (column_O-column)/stepSize+1;
      printf("scanning %d x %d steps\n", scanx, scany);
      cudaMalloc((void**)&objectWave, row_O*column_O*sizeof(Real)*2);
      cudaMalloc((void**)&pupilpatternWave, sz*2);
      cudaMalloc((void**)&esw, sz*2);
      cudaMalloc((void**)&complexCache, sz*2);
      patterns = (Real**) malloc(scanx*scany*sizeof(Real*));
      for(int i = 0; i < scanx*scany ; i++) {
        cudaMalloc((void**)&(patterns[i]),sz);
      }
      printf("initializing cuda image\n");
      init_cuda_image(row_O,column_O,rcolor,1./exposure);
      cudaF(initRand)(devstates);
    }
    void readPupilAndObject(){
      Real* object_intensity = readImage(common.Intensity.c_str(), row_O, column_O);
      Real* object_phase = readImage(common.Phase.c_str(), row_O, column_O);
      int objsz = row_O*column_O*sizeof(Real);
      Real* d_object_intensity;
      Real* d_object_phase;
      cudaMalloc((void**)&d_object_intensity, objsz);
      cudaMalloc((void**)&d_object_phase, objsz);
      cudaMemcpy(d_object_intensity, object_intensity, objsz, cudaMemcpyHostToDevice);
      cudaMemcpy(d_object_phase, object_phase, objsz, cudaMemcpyHostToDevice);
      free(object_intensity);
      free(object_phase);
      Real* pupil_intensity = readImage(pupil.Intensity.c_str(), row, column);
      sz = row*column*sizeof(Real);
      int row_tmp=row*oversampling;
      int column_tmp=column*oversampling;
      allocateMem();
      cudaF(createWaveFront)(d_object_intensity, d_object_phase, (complexFormat*)objectWave, 1);
      cudaFree(d_object_intensity);
      cudaFree(d_object_phase);
      verbose(2,
          plt.init(row_O,column_O);
          plt.plotComplex(objectWave, MOD2, 0, 1, "inputObject");
      )
      Real* d_intensity = patterns[0]; //use the memory allocated;
      cudaMemcpy(d_intensity, pupil_intensity, sz, cudaMemcpyHostToDevice);
      free(pupil_intensity);
      Real* d_phase = 0;
      if(doPhaseModulationPupil){
        d_phase = patterns[1];
        int tmp;
        Real* pupil_phase = readImage(pupil.Phase.c_str(), tmp,tmp);
        gpuErrchk(cudaMemcpy(d_phase, pupil_phase, sz, cudaMemcpyHostToDevice));
        free(pupil_phase);
      }
      cudaMalloc((void**)&pupilobjectWave, row_tmp*column_tmp*sizeof(complexFormat));
      init_cuda_image(row_tmp,column_tmp,rcolor, 1./exposure);
      cudaF(createWaveFront)(d_intensity, d_phase, (complexFormat*)pupilobjectWave, oversampling);
      plt.init(row_tmp,column_tmp);
      plt.plotComplex(pupilobjectWave, MOD2, 0, 1, "pupilIntensity", 0);
      opticalPropagate((complexFormat*)pupilobjectWave, lambda, dpupil, beamspotsize*oversampling, row_tmp, column_tmp);
      init_cuda_image(row,column,rcolor, 1./exposure);
      cudaF(crop)((complexFormat*)pupilobjectWave, (complexFormat*)pupilpatternWave, row_tmp, column_tmp);
      plt.init(row,column);
      plt.plotComplex(pupilpatternWave, MOD2, 0, 1, "probeIntensity", 1);
      plt.plotComplex(pupilpatternWave, PHASE, 0, 1, "probePhase", 0);
      calculateParameters();
      multiplyFresnelPhase(pupilpatternWave, d);
    }
    void createPattern(){
      int idx = 0;
      if(useBS) {
        createBeamStop();
        plt.plotFloat(beamstop, MOD, 1, 1,"beamstop", 0);
      }
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          cudaF(multiplyProbe)((complexFormat*)objectWave, (complexFormat*)pupilpatternWave, esw, i*stepSize, j*stepSize, row_O, column_O);
          verbose(5, plt.plotComplex(esw, MOD2, 0, 1, ("ptycho_esw"+to_string(i)+"_"+to_string(j)).c_str()));
          propagate(esw,esw,1);
          cudaF(getMod2)(patterns[idx], esw);
          cudaF(applySupport)(patterns[idx], beamstop);
          if(simCCDbit) cudaF(applyPoissonNoise_WO)(patterns[idx], noiseLevel, devstates, 1./exposure);
          verbose(3, plt.plotFloat(patterns[idx], MOD, 1, exposure, (common.Pattern+to_string(i)+"_"+to_string(j)).c_str()));
          verbose(4, plt.plotFloat(patterns[idx], MOD, 1, exposure, (common.Pattern+to_string(i)+"_"+to_string(j)+"log").c_str(),1));
          idx++;
        }
      }
    }
    void initObject(){
      init_cuda_image(row_O,column_O,rcolor, 1./exposure);
      cudaF(random)((complexFormat*)objectWave, devstates);
      init_cuda_image(row,column,rcolor, 1./exposure);
      cudaF(pupilFunc)((complexFormat*)pupilpatternWave);
      //cudaF(random)((complexFormat*)pupilpatternWave, devstates);
    }
    void iterate(){
      init_cuda_image(row,column,rcolor, 1./exposure);
      Real objMax;
      for(int iter = 0; iter < nIter; iter++){
        int idx = 0;
        for(int i = 0; i < scanx; i++){
          for(int j = 0; j < scany; j++){
            cudaF(multiplyProbe)((complexFormat*)objectWave, (complexFormat*)pupilpatternWave, esw,
              i*stepSize, j*stepSize, row_O, column_O, complexCache);
            if(iter >= 4) objMax = findMax((complexFormat*)complexCache, row*column);
            Real probeMax = findMax((complexFormat*)pupilpatternWave, row*column);
            propagate(esw,complexCache,1);
            cudaF(applyMod)(complexCache, patterns[idx],beamstop,1);
            propagate(complexCache,complexCache,0);
            cudaF(add)(esw, complexCache, -1);
            if(iter < 4) cudaF(updateObject)((complexFormat*)objectWave, (complexFormat*)pupilpatternWave, esw,
                i*stepSize, j*stepSize, row_O, column_O,//1,1);
                probeMax);
            else cudaF(updateObjectAndProbe)((complexFormat*)objectWave, (complexFormat*)pupilpatternWave, esw,
                i*stepSize, j*stepSize, row_O, column_O,//1,1);
                probeMax, objMax);
            idx++;
          }
        }
      }
      plt.init(row, column);
      plt.plotComplex(pupilpatternWave, MOD2, 0, 1, "ptycho_probe_afterIter", 0);
      init_cuda_image(row_O,column_O,rcolor,1./exposure);
      plt.init(row_O, column_O);
      plt.plotComplex(objectWave, MOD2, 0, 1, "ptycho_afterIter", 0);
      plt.plotComplex(objectWave, PHASE, 0, 1, "ptycho_afterIterphase", 0);
    }
    void readPattern(){
      Real* pattern = readImage((common.Pattern+"0_0.png").c_str(), row, column);
      plt.init(row,column);
      sz = row*column*sizeof(Real);
      allocateMem();
      init_cuda_image(row,column,rcolor, 1./exposure);
      createBeamStop();
      int idx = 0;
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          if(idx!=0) pattern = readImage((common.Pattern+to_string(i)+"_"+to_string(j)+".png").c_str(), row, column);
          cudaMemcpy(patterns[idx], pattern, sz, cudaMemcpyHostToDevice);
          free(pattern);
          cudaF(cudaConvertFO)(patterns[idx]);
          cudaF(applyNorm)(patterns[idx], 1./exposure);
          verbose(3, plt.plotFloat(patterns[idx], MOD, 1, exposure, ("input"+common.Pattern+to_string(i)+"_"+to_string(j)).c_str()));
          idx++;
        }
      }
      printf("Created pattern data\n");
      calculateParameters();
    }
    void calculateParameters(){
      resolution = lambda*dpupil/beamspotsize/oversampling;
      if(runSim) d = resolution*pixelsize*row/lambda;
      experimentConfig::calculateParameters();
    }
};
int main(int argc, char** argv )
{
  ptycho setups(argv[1]);
  cudaFree(0); // to speed up the cudaMalloc; https://forums.developer.nvidia.com/t/cudamalloc-slow/40238
  if(argc < 2){
    printf("please feed the object intensity and phase image\n");
  }
  //setups.readPupilAndObject();
  //setups.createPattern();
  setups.readPattern();
  printf("Imaging distance = %4.2fcm\n", setups.d*1e-4);
  printf("fresnel factor = %f\n", setups.fresnelFactor);
  printf("enhancement = %f\n", setups.enhancement);

  printf("pupil Imaging distance = %4.2fcm\n", setups.dpupil*1e-4);
  printf("pupil fresnel factor = %f\n", setups.fresnelFactorpupil);
  printf("pupil enhancement = %f\n", setups.enhancementpupil);
  setups.initObject();
  setups.iterate();

  return 0;
}

