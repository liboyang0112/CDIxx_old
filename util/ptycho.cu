#include <complex>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <random>
#include <chrono>

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
#define DELTA 1e-3
#define GAMMA 0.5
#define cudaIdx() \
int x = blockIdx.x * blockDim.x + threadIdx.x;\
int y = blockIdx.y * blockDim.y + threadIdx.y;\
if(x >= cuda_row || y >= cuda_column) return;\
int index = x*cuda_column + y;

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
  d_out = (complexFormat*)memMngr.borrowCache(sizeof(complexFormat));

  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;
  CustomMax max_op;
  complexFormat tmp;
  tmp.x = tmp.y = 0;
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, tmp));
  d_temp_storage = memMngr.borrowCache(temp_storage_bytes);

  // Run
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, tmp));
  complexFormat output;
  cudaMemcpy(&output, d_out, sizeof(complexFormat), cudaMemcpyDeviceToHost);

  if (d_out) memMngr.returnCache(d_out);
  if (d_temp_storage) memMngr.returnCache(d_temp_storage);
  return output.x*output.x + output.y*output.y;
}

struct CustomSumReal
{
  __device__ __forceinline__
    complexFormat operator()(const complexFormat &a, const complexFormat &b) const {
      return {a.x+b.x,0};
    }
};

Real findSumReal(complexFormat* d_in, int num_items)
{
  complexFormat *d_out = NULL;
  d_out = (complexFormat*)memMngr.borrowCache(sizeof(complexFormat));

  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;
  CustomSumReal sum_op;
  complexFormat tmp;
  tmp.x = 0;
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, sum_op, tmp));
  d_temp_storage = memMngr.borrowCache(temp_storage_bytes);

  // Run
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, sum_op, tmp));
  complexFormat output;
  cudaMemcpy(&output, d_out, sizeof(complexFormat), cudaMemcpyDeviceToHost);

  if (d_out) memMngr.returnCache(d_out);
  if (d_temp_storage) memMngr.returnCache(d_temp_storage);
  return output.x;
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
  cudaIdx();
  if(support[index] > cuda_threshold) image[index] = 0;
}


__global__ void multiplyProbe(complexFormat* object, complexFormat* probe, complexFormat* U, int shiftx, int shifty, int objrow, int objcol, complexFormat *window = 0){
  cudaIdx();
  complexFormat tmp;
  if(x+shiftx >= objrow || y+shifty >= objcol || x+shiftx < 0 || y+shifty < 0) tmp.x = tmp.y = 0;
  else tmp =  object[(x+shiftx)*objcol+y+shifty];
  if(window) window[index] = tmp;
  U[index] = cuCmulf(probe[index], tmp);
}

__global__ void getWindow(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat *window){
  cudaIdx();
  complexFormat tmp;
  if(x+shiftx >= objrow || y+shifty >= objcol || x+shiftx < 0 || y+shifty < 0) tmp.x = tmp.y = 0;
  else tmp =  object[(x+shiftx)*objcol+y+shifty];
  window[index] = tmp;
}

__global__ void updateWindow(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat *window){
  cudaIdx();
  if(x+shiftx >= objrow || y+shifty >= objcol || x+shiftx < 0 || y+shifty < 0) return;
  object[(x+shiftx)*objcol+y+shifty] = window[index];
}


__device__ void ePIE(complexFormat &target, complexFormat source, complexFormat &diff, Real maxi, Real param){
  Real denom = param/(maxi);
  source = cuCmulf(cuConjf(source),diff);
  target.x -= source.x*denom;
  target.y -= source.y*denom;
}

__device__ void rPIE(complexFormat &target, complexFormat source, complexFormat &diff, Real maxi, Real param){
  Real denom = source.x*source.x+source.y*source.y;
//  if(denom < 8e-4*maxi) return;
  denom = 1./((1-param)*denom+param*maxi);
  source = cuCmulf(cuConjf(source),diff);
  target.x -= source.x*denom;
  target.y -= source.y*denom;
}

__global__ void updateObject(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe){
  cudaIdx()
  rPIE(object[index], probe[index], U[index], mod2maxProbe, ALPHA);
}

__global__ void updateObjectAndProbe(complexFormat* object, complexFormat* probe, complexFormat* U, Real mod2maxProbe, Real mod2maxObj){
  cudaIdx()
  complexFormat objectdat= object[index];
  complexFormat diff= U[index];
  rPIE(object[index], probe[index], diff, mod2maxProbe, ALPHA);
  rPIE(probe[index], objectdat, diff, mod2maxObj, BETA);
}

__global__ void random(complexFormat* object, curandStateMRG32k3a *state){
  cudaIdx()
  curand_init(1,index,0,&state[index]);
  object[index].x = curand_uniform(&state[index]);
  object[index].y = curand_uniform(&state[index]);
}

__global__ void pupilFunc(complexFormat* object){
  cudaIdx()
  int shiftx = x - cuda_row/2;
  int shifty = y - cuda_column/2;
  object[index].x = 3*gaussian(shiftx,shifty,cuda_row/8);
  object[index].y = 0;
}

__global__ void multiplyShift(complexFormat* object, Real shiftx, Real shifty){
  cudaIdx();
  Real phi = -2*M_PI*(shiftx*(x-cuda_row/2)/cuda_row+shifty*(y-cuda_column/2)/cuda_column);
  complexFormat tmp = {cos(phi),sin(phi)};
  object[index] = cuCmulf(object[index],tmp);
}

__global__ void multiplyx(complexFormat* object){
  cudaIdx();
  //object[index].x *= Real(x)/cuda_row;
  //object[index].y *= Real(x)/cuda_row;
  object[index].x *= Real(x)/cuda_row-0.5;
  object[index].y *= Real(x)/cuda_row-0.5;
}

__global__ void multiplyy(complexFormat* object){
  cudaIdx();
  //object[index].x *= Real(y)/cuda_row;
  //object[index].y *= Real(y)/cuda_row;
  object[index].x *= Real(y)/cuda_row-0.5;
  object[index].y *= Real(y)/cuda_row-0.5;
}

__global__ void calcPartial(complexFormat* object, complexFormat* Fn, Real* pattern, Real* beamstop){
  cudaIdx();
  if(beamstop[index] > 0.5){
    object[index].x = 0;
    return;
  }
  Real ret;
  auto fntmp = Fn[index];
  Real fnmod2 = fntmp.x*fntmp.x + fntmp.y*fntmp.y;
  ret = fntmp.x*object[index].y - fntmp.y*object[index].x;
  Real fact = pattern[index]+DELTA;
  if(fact<0) fact = 0;
  /*
  if(pattern[index]+DELTA<0) fact = 0;
  else fact = pow(pattern[index]+DELTA,GAMMA);
  ret*=(pow(fnmod2+DELTA,GAMMA)-fact);
  ret*=GAMMA*pow(fnmod2+DELTA,GAMMA-1);
  */
  ret*=1-sqrt(fact/(fnmod2+DELTA));
  //if(ret>1) printf("FIND larget ret %f at (%d, %d): (%f, %f), (%f, %f), %f, %f\n",ret, x, y, object[index].x, object[index].y, fntmp.x, fntmp.y, pattern[index], beamstop[index]);
  object[index].x = ret;
}

void shiftWave(complexFormat* wave, int npix, Real shiftx, Real shifty){
  myCufftExec( *plan, wave, wave, CUFFT_FORWARD);
  cudaF(cudaConvertFO)(wave);
  cudaF(multiplyShift)(wave, shiftx, shifty);
  cudaF(cudaConvertFO)(wave);
  myCufftExec( *plan, wave, wave, CUFFT_INVERSE);
  cudaF(applyNorm)(wave, 1./npix);
}

class ptycho : public experimentConfig{
  public:
    int row_O = 512;  //in ptychography this is different from row (the size of probe).
    int column_O = 512;
    int sz = 0;
    int stepSize = 32;
    int doPhaseModulationPupil = 0;
    int scanx = 0;
    int scany = 0;
    Real *shiftx = 0;
    Real *shifty = 0;
    Real **patterns; //patterns[i*scany+j] points to the address on device to store pattern;
    complexFormat *esw;
    curandStateMRG32k3a *devstates = 0;

    ptycho(const char* configfile):experimentConfig(configfile){}
    void allocateMem(){
      if(devstates) return;
      devstates = (curandStateMRG32k3a*) memMngr.borrowCache(column_O * row_O * sizeof(curandStateMRG32k3a));
      printf("allocating memory\n");
      scanx = (row_O-row)/stepSize+1;
      scany = (column_O-column)/stepSize+1;
      printf("scanning %d x %d steps\n", scanx, scany);
      objectWave = (complexFormat*)memMngr.borrowCache(row_O*column_O*sizeof(Real)*2);
      pupilpatternWave = (complexFormat*)memMngr.borrowCache(sz*2);
      esw = (complexFormat*) memMngr.borrowCache(sz*2);
      patterns = (Real**) malloc(scanx*scany*sizeof(Real*));
      memset(patterns, 0, scanx*scany*sizeof(Real*)/sizeof(char));
      printf("initializing cuda image\n");
      init_cuda_image(row_O,column_O,rcolor,1./exposure);
      cudaF(initRand)(devstates);
      if(positionUncertainty > 1e-4){
        initPosition();
      }
    }
    void readPupilAndObject(){
      Real* object_intensity = readImage(common.Intensity.c_str(), row_O, column_O);
      Real* object_phase = readImage(common.Phase.c_str(), row_O, column_O);
      int objsz = row_O*column_O*sizeof(Real);
      Real* d_object_intensity;
      Real* d_object_phase;
      d_object_intensity = (Real*)memMngr.borrowCache(objsz);
      d_object_phase = (Real*)memMngr.borrowCache(objsz);
      cudaMemcpy(d_object_intensity, object_intensity, objsz, cudaMemcpyHostToDevice);
      cudaMemcpy(d_object_phase, object_phase, objsz, cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(object_intensity);
      ccmemMngr.returnCache(object_phase);
      Real* pupil_intensity = readImage(pupil.Intensity.c_str(), row, column);
      sz = row*column*sizeof(Real);
      int row_tmp=row*oversampling;
      int column_tmp=column*oversampling;
      allocateMem();
      cudaF(createWaveFront)(d_object_intensity, d_object_phase, (complexFormat*)objectWave, 1);
      memMngr.returnCache(d_object_intensity);
      memMngr.returnCache(d_object_phase);
      verbose(2,
          plt.init(row_O,column_O);
          plt.plotComplex(objectWave, MOD2, 0, 1, "inputObject");
          plt.plotComplex(objectWave, PHASE, 0, 1, "inputPhase");
          //plt.plotPhase(objectWave, PHASERAD, 0, 1, "inputPhase");
      )
      Real* d_intensity = (Real*) memMngr.borrowCache(sz); //use the memory allocated;
      cudaMemcpy(d_intensity, pupil_intensity, sz, cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(pupil_intensity);
      Real* d_phase = 0;
      if(doPhaseModulationPupil){
        d_phase = (Real*) memMngr.borrowCache(sz);
        int tmp;
        Real* pupil_phase = readImage(pupil.Phase.c_str(), tmp,tmp);
        gpuErrchk(cudaMemcpy(d_phase, pupil_phase, sz, cudaMemcpyHostToDevice));
        ccmemMngr.returnCache(pupil_phase);
      }
      pupilobjectWave = (complexFormat*)memMngr.borrowCache(row_tmp*column_tmp*sizeof(complexFormat));
      init_cuda_image(row_tmp,column_tmp,rcolor, 1./exposure);
      cudaF(createWaveFront)(d_intensity, d_phase, (complexFormat*)pupilobjectWave, oversampling);
      memMngr.returnCache(d_intensity);
      if(d_phase) memMngr.returnCache(d_phase);
      plt.init(row_tmp,column_tmp);
      plt.plotComplex(pupilobjectWave, MOD2, 0, 1, "pupilIntensity", 0);
      init_fft(row_tmp,column_tmp);
      opticalPropagate((complexFormat*)pupilobjectWave, lambda, dpupil, beamspotsize*oversampling, row_tmp, column_tmp);
      plt.plotComplex(pupilobjectWave, MOD2, 0, 1, "pupilPattern", 0);
      init_cuda_image(row,column,rcolor, 1./exposure);
      init_fft(row,column);
      cudaF(crop)((complexFormat*)pupilobjectWave, (complexFormat*)pupilpatternWave, row_tmp, column_tmp);
      plt.init(row,column);
      plt.plotComplex(pupilpatternWave, MOD2, 0, 1, "probeIntensity", 0);
      plt.plotComplex(pupilpatternWave, PHASE, 0, 1, "probePhase", 0);
      calculateParameters();
      multiplyFresnelPhase(pupilpatternWave, d);
    }
    void initPosition(){
      shiftx = (Real*)ccmemMngr.borrowCache(scanx*scany*sizeof(Real));
      shifty = (Real*)ccmemMngr.borrowCache(scanx*scany*sizeof(Real));
      memset(shiftx, 0, scanx*scany*sizeof(Real)/sizeof(char));
      memset(shifty, 0, scanx*scany*sizeof(Real)/sizeof(char));
      if(runSim && positionUncertainty>1e-4){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::normal_distribution<double> distribution(0.0, 1.);
        for(int i = 0 ; i < scanx*scany; i++){
          shiftx[i]+= distribution(generator)*positionUncertainty;
          shifty[i]+= distribution(generator)*positionUncertainty;
          printf("shifts (%d, %d): (%f, %f)\n", i/scany, i%scany, shiftx[i],shifty[i]);
        }
      }
    }
    void resetPosition(){
      for(int i = 0 ; i < scanx*scany; i++){
        shiftx[i] = shifty[i] = 0;
      }
    }
    void createPattern(){
      int idx = 0;
      if(useBS) {
        createBeamStop();
        plt.plotFloat(beamstop, MOD, 1, 1,"beamstop", 0);
      }
      complexFormat* window = (complexFormat*)memMngr.borrowCache(sz*2);
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          int shiftxpix = shiftx[idx]-round(shiftx[idx]);
          int shiftypix = shiftx[idx]-round(shifty[idx]);
          cudaF(getWindow)((complexFormat*)objectWave, i*stepSize-round(shiftx[idx]), j*stepSize-round(shifty[idx]), row_O, column_O, window);
          if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
            shiftWave((complexFormat*)window, row*column, shiftxpix, shiftypix);
          }
          cudaF(multiply)(esw, (complexFormat*)pupilpatternWave, window);
          verbose(5, plt.plotComplex(esw, MOD2, 0, 1, ("ptycho_esw"+to_string(i)+"_"+to_string(j)).c_str()));
          propagate(esw,esw,1);
          if(!patterns[idx]) patterns[idx] = (Real*)memMngr.borrowCache(sz);
          cudaF(getMod2)(patterns[idx], esw);
          if(useBS) cudaF(applySupport)(patterns[idx], beamstop);
          if(simCCDbit) cudaF(applyPoissonNoise_WO)(patterns[idx], noiseLevel, devstates, 1./exposure);
          verbose(3, plt.plotFloat(patterns[idx], MOD, 1, exposure, (common.Pattern+to_string(i)+"_"+to_string(j)).c_str()));
          verbose(4, plt.plotFloat(patterns[idx], MOD, 1, exposure, (common.Pattern+to_string(i)+"_"+to_string(j)+"log").c_str(),1));
          idx++;
        }
      }
      memMngr.returnCache(window);
    }
    void initObject(){
      init_cuda_image(row_O,column_O,rcolor, 1./exposure);
      cudaF(random)((complexFormat*)objectWave, devstates);
      init_cuda_image(row,column,rcolor, 1./exposure);
      cudaF(pupilFunc)((complexFormat*)pupilpatternWave);
    }
    void updatePosition(Real &shiftx, Real &shifty, complexFormat* obj, complexFormat* probe, Real* pattern, complexFormat* Fn){
      Real siz = memMngr.getSize(obj);
      complexFormat *cachex = (complexFormat*)memMngr.borrowCache(siz);
      complexFormat *cachey = (complexFormat*)memMngr.borrowCache(siz);
      propagate(obj, cachex, 1);
      cudaF(cudaConvertFO)(cachex);
      cudaMemcpy(cachey, cachex, siz, cudaMemcpyDeviceToDevice);
      cudaF(multiplyx)(cachex);
      cudaF(multiplyy)(cachey);
      cudaF(cudaConvertFO)(cachex);
      cudaF(cudaConvertFO)(cachey);
      propagate(cachex, cachex, 0);
      propagate(cachey, cachey, 0);
      cudaF(multiply)(cachex, probe);
      cudaF(multiply)(cachey, probe);
      propagate(cachex, cachex, 1);
      propagate(cachey, cachey, 1);
      cudaF(calcPartial)(cachex, Fn, pattern, beamstop);
      cudaF(calcPartial)(cachey, Fn, pattern, beamstop);
      Real partialx = 8*M_PI*findSumReal(cachex,row*column);
      Real partialy = 8*M_PI*findSumReal(cachey,row*column);
      memMngr.returnCache(cachex);
      memMngr.returnCache(cachey);
      shiftx -= partialx*1e-3;
      shifty -= partialy*1e-3;
      if(shiftx!=shiftx || shifty!=shifty) exit(0);
    }
    void iterate(){
      resetPosition();
      init_cuda_image(row,column,rcolor, 1./exposure);
      Real objMax;
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2);
      complexFormat *objCache = (complexFormat*)memMngr.borrowCache(sz*2);
      int update_probe_iter = 4;
      for(int iter = 0; iter < nIter; iter++){
        int idx = 0;
        for(int i = 0; i < scanx; i++){
          for(int j = 0; j < scany; j++){
            int shiftxpix = shiftx[idx]-round(shiftx[idx]);
            int shiftypix = shiftx[idx]-round(shifty[idx]);
            cudaF(getWindow)((complexFormat*)objectWave,
                i*stepSize-round(shiftx[idx]), j*stepSize-round(shifty[idx]), row_O, column_O, objCache);
            if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
              shiftWave((complexFormat*)objCache, row*column, shiftxpix, shiftypix);
            }
            cudaF(multiply)(esw, (complexFormat*)pupilpatternWave, objCache);
            if(iter >= update_probe_iter) objMax = findMax((complexFormat*)objCache, row*column);
            propagate(esw,Fn,1);
            if(iter % 100 == 0 && iter >= 100) updatePosition(shiftx[idx], shifty[idx], objCache, (complexFormat*)pupilpatternWave, patterns[idx], Fn);
            //if(iter > 150 && i == 4 && j == 4) updatePosition(shiftx[idx], shifty[idx], objCache, (complexFormat*)pupilpatternWave, patterns[idx], Fn);
            Real probeMax = findMax((complexFormat*)pupilpatternWave, row*column);
            cudaF(applyMod)(Fn, patterns[idx],beamstop,1);
            propagate(Fn,Fn,0);
            cudaF(add)(esw, Fn, -1);
            if(iter < update_probe_iter) cudaF(updateObject)(objCache, (complexFormat*)pupilpatternWave, esw,//1,1);
                probeMax);
            else cudaF(updateObjectAndProbe)(objCache, (complexFormat*)pupilpatternWave, esw,//1,1);
                probeMax, objMax);
            if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
              shiftWave(objCache, row*column, -shiftxpix, -shiftypix);
            }
            cudaF(updateWindow)((complexFormat*)objectWave,
                i*stepSize-round(shiftx[idx]), j*stepSize-round(shifty[idx]), row_O, column_O, objCache);
            idx++;
          }
        }
        if(iter == 200){
          init_cuda_image(row_O,column_O,rcolor,1./exposure);
          plt.init(row_O, column_O);
          plt.plotComplex(objectWave, MOD2, 0, 1, "ptycho_b4position", 0);
          plt.plotComplex(objectWave, PHASE, 0, 1, "ptycho_b4positionphase", 0);
          init_cuda_image(row,column,rcolor,1./exposure);
          plt.init(row, column);
        }
      }
      for(int i = 0 ; i < scanx*scany; i++){
        printf("recon shifts (%d, %d): (%f, %f)\n", i/scany, i%scany, shiftx[i],shifty[i]);
      }
      memMngr.returnCache(Fn);
      memMngr.returnCache(objCache);
      plt.init(row, column);
      plt.plotComplex(pupilpatternWave, MOD2, 0, 1, "ptycho_probe_afterIter", 0);
      init_cuda_image(row_O*4/7,column_O*4/7,rcolor,1./exposure);
      plt.init(row_O*4/7, column_O*4/7);
      complexFormat* output = (complexFormat*)memMngr.borrowCache(row_O*2/3*(column_O*4/7)*sizeof(complexFormat));
      cudaF(crop)((complexFormat*)objectWave, output, row_O, column_O);
      plt.plotComplex(output, MOD2, 0, 1.5/findMax(output, row_O*4/7*(column_O*4/7)), "ptycho_afterIter", 0);
      //plt.plotComplex(objectWave, PHASE, 0, 1, "ptycho_afterIterphase", 0);
      plt.plotPhase(output, PHASERAD, 0, 1, "ptycho_afterIterphase", 0);
    }
    void readPattern(){
      Real* pattern = readImage((common.Pattern+"0_0.png").c_str(), row, column);
      plt.init(row,column);
      init_fft(row,column);
      sz = row*column*sizeof(Real);
      allocateMem();
      init_cuda_image(row,column,rcolor, 1./exposure);
      createBeamStop();
      int idx = 0;
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          if(idx!=0) pattern = readImage((common.Pattern+to_string(i)+"_"+to_string(j)+".png").c_str(), row, column);
          if(!patterns[idx]) patterns[idx] = (Real*)memMngr.borrowCache(sz);
          cudaMemcpy(patterns[idx], pattern, sz, cudaMemcpyHostToDevice);
          ccmemMngr.returnCache(pattern);
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
  if(setups.runSim){
    setups.readPupilAndObject();
    setups.createPattern();
  }else{
    setups.readPattern();
  }
  printf("Imaging distance = %4.2fcm\n", setups.d*1e-4);
  printf("fresnel factor = %f\n", setups.fresnelFactor);
  printf("Resolution = %4.2fnm\n", setups.resolution*1e3);

  printf("pupil Imaging distance = %4.2fcm\n", setups.dpupil*1e-4);
  printf("pupil fresnel factor = %f\n", setups.fresnelFactorpupil);
  printf("pupil enhancement = %f\n", setups.enhancementpupil);
  setups.initObject();
  setups.iterate();

  return 0;
}

