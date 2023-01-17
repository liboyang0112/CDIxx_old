#include <iostream>
#include <fstream>
#include <vector>
#include "cudaConfig.h"
#include "cuPlotter.h"
#include "mnistData.h"
#include "common.h"
#include "monoChromo.h"

int main(int argc, char** argv){
  if(argc==1) { printf("Tell me which one is the mnist data folder\n"); }
  monoChromo mwl(argv[1]);
  cuMnist *mnist_dat = 0;
  int objrow;
  int objcol;
  Real* d_input;
  Real* intensity;
  if(mwl.runSim){
    if(mwl.domnist) {
      objrow = 128;
      objcol = 128;
      mnist_dat = new cuMnist(mwl.mnistData.c_str(), 3, objrow, objcol);
      cudaMalloc((void**)&d_input, objrow*objcol*sizeof(Real));
    }
    else {
      intensity = readImage(mwl.common.Intensity.c_str(), objrow, objcol);
      cudaMalloc((void**)&d_input, objrow*objcol*sizeof(Real));
      cudaMemcpy(d_input, intensity, objrow*objcol*sizeof(Real), cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(intensity);
    }
  }else{
    intensity = readImage(mwl.common.Pattern.c_str(), objrow, objcol);
    objrow/=mwl.oversampling;
    objcol/=mwl.oversampling;
    ccmemMngr.returnCache(intensity);
  }
#if 0
  int lambdarange = 4;
  int nlambda = mwl.oversampling*objrow*(lambdarange-1)/2;
  Real *lambdas;
  Real *spectra;
  lambdas = (Real*)ccmemMngr.borrowCache(sizeof(Real)*nlambda);
  spectra = (Real*)ccmemMngr.borrowCache(sizeof(Real)*nlambda);
  std::ofstream spectrafile;
  spectrafile.open("spectra.txt",ios::out);
  for(int i = 0; i < nlambda; i++){
    lambdas[i] = 1 + 2.*i/objrow/mwl.oversampling;
    spectra[i] = exp(-pow(i*2./nlambda-1,2))/nlambda; //gaussian, -1,1 with sigma=1
  }
#elif 0
  const int nlambda = 5;
  Real lambdas[nlambda] = {1, 11./9, 11./7, 11./5, 11./3};
  Real spectra[nlambda] = {0.1,0.2,0.3,0.3,0.1};
#else
  int nlambda;
  Real* lambdas, *spectra;
  Real startlambda = 700;
  Real endlambda = 1000;
  Real rate = endlambda/startlambda;
  getNormSpectrum(argv[2],argv[3],startlambda,nlambda,lambdas,spectra); //this may change startlambda
  printf("lambda range = (%f, %f), ratio=%f", startlambda, endlambda, rate);
  rate = 1.15;
#endif
  std::ofstream spectrafile;
  spectrafile.open("spectra.txt",ios::out);
  for(int i = 0; i < nlambda; i++){
    spectrafile<<lambdas[i]<<" "<<spectra[i]<<endl;
  }
  spectrafile.close();
  //mwl.init(objrow, objcol, nlambda, lambdas, spectra);
  mwl.init(objrow, objcol, lambdas, spectra, rate);
  int sz = mwl.row*mwl.column*sizeof(Real);
  Real *d_patternSum = (Real*)memMngr.borrowCache(sz);
  complexFormat *single = (complexFormat*)memMngr.borrowCache(sz*2);
  complexFormat *d_CpatternSum = (complexFormat*)memMngr.borrowCache(sz*2);
  complexFormat *d_solved = (complexFormat*)memMngr.borrowCache(sz*2);
  init_cuda_image(mwl.row, mwl.column);
  mwl.plt.init(mwl.row, mwl.column);
  curandStateMRG32k3a *devstates = (curandStateMRG32k3a *)memMngr.borrowCache(mwl.column * mwl.row * sizeof(curandStateMRG32k3a));
  cudaF(initRand)(devstates);
  for(int j = 0; j < 1; j++){
    if(mwl.runSim && mwl.domnist) {
        mnist_dat->cuRead(d_input);
        init_cuda_image(objrow, objcol);
        mwl.plt.init(objrow, objcol);
        mwl.plt.plotFloat(d_input, MOD, 0, 1, ("input"+to_string(j)).c_str(), 0);
        init_cuda_image(mwl.row, mwl.column);
        mwl.plt.init(mwl.row, mwl.column);
    }
    if(mwl.runSim){
      mwl.generateMWL(d_input, d_patternSum, single);
      cudaF(applyPoissonNoise_WO)(d_patternSum, mwl.noiseLevel, devstates);
      mwl.plt.plotFloat(d_patternSum, MOD, 0, mwl.exposure*2, ("merged"+to_string(j)).c_str(), 0);
      mwl.plt.plotFloat(single, MOD, 0, mwl.exposure, ("single"+to_string(j)).c_str(), 1);
    }
    intensity = readImage(mwl.common.Pattern.c_str(), objrow, objcol);
    cudaMemcpy(d_patternSum, intensity, mwl.row*mwl.column*sizeof(Real), cudaMemcpyHostToDevice);
    cudaF(extendToComplex)(d_patternSum, d_CpatternSum);
    mwl.plt.plotFloat(d_patternSum, MOD, 0, 1, ("mergedlog"+to_string(j)).c_str(), 1);
    //mwl.solveMWL(d_CpatternSum, d_solved, single);
    mwl.solveMWL(d_CpatternSum, d_solved);
    mwl.plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(j)).c_str(), 0);
    mwl.plt.plotComplex(d_solved, MOD, 0, 1, ("solvedlog"+to_string(j)).c_str(), 1);
    myCufftExec(*plan, d_solved, d_CpatternSum, CUFFT_FORWARD);
    mwl.plt.plotComplex(d_CpatternSum, MOD, 1, 2./mwl.row, ("autocsolved"+to_string(j)).c_str(), 1);
  }

  return 0;
}

