#include <iostream>
#include <vector>
#include "cudaConfig.h"
#include "cuPlotter.h"
#include "mnistData.h"
#include "common.h"
#include "monoChromo.h"

int main(int argc, char** argv){
  if(argc==1) { printf("Tell me which one is the mnist data folder\n"); }
  Real lambdas[] = {1,11./9,11./7, 11./5, 11./3};
  Real spectra[] = {0.1,0.4,0.3,0.2, 0.1};
  monoChromo mwl(argv[1], 5, lambdas, spectra);
  cuMnist *mnist_dat = 0;
  int objrow = 128;
  int objcol = 128;
  Real* d_input;
  if(mwl.domnist) {
    mnist_dat = new cuMnist(mwl.mnistData.c_str(), 3, objrow, objcol);
    cudaMalloc((void**)&d_input, objrow*objcol*sizeof(Real));
  }
  else {
    Real* intensity = readImage(mwl.common.Intensity.c_str(), objrow, objcol);
    cudaMalloc((void**)&d_input, objrow*objcol*sizeof(Real));
    cudaMemcpy(d_input, intensity, objrow*objcol*sizeof(Real), cudaMemcpyHostToDevice);
  }
  mwl.init(objrow,objcol);
  Real *d_patternSum;
  complexFormat *single = (complexFormat*)memMngr.borrowCache(mwl.row*mwl.column*sizeof(Real)*2);
  complexFormat *d_CpatternSum;
  Real *d_solved;
  cudaMalloc((void**)&d_patternSum, mwl.row*mwl.column*sizeof(Real));
  cudaMalloc((void**)&d_CpatternSum, mwl.row*mwl.column*sizeof(Real)*2);
  cudaMalloc((void**)&d_solved, mwl.row*mwl.column*sizeof(Real)*2);
  for(int j = 0; j < mwl.domnist?3000:1; j++){
    if(mwl.domnist) mnist_dat->cuRead(d_input);
    mwl.generateMWL(d_input, d_patternSum, single);
    cudaF(extendToComplex)(d_patternSum, d_CpatternSum);
    //mwl.solveMWL(d_CpatternSum, d_solved, single);
    mwl.solveMWL(d_CpatternSum, d_solved);
    mwl.plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
    mwl.plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(j)).c_str(), 0);
  }

  return 0;
}

