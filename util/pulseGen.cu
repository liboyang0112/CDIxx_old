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
  Real *lambdas;
  Real *spectra;
  monoChromo mwl(argv[1]);
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
  int lambdarange = 2;
  int nlambda = mwl.oversampling*objrow*(lambdarange-1)/2;
  lambdas = (Real*)ccmemMngr.borrowCache(sizeof(Real)*nlambda);
  spectra = (Real*)ccmemMngr.borrowCache(sizeof(Real)*nlambda);
  std::ofstream spectrafile;
  spectrafile.open("spectra.txt",ios::out);
  for(int i = 0; i < nlambda; i++){
    lambdas[i] = 1 + 2.*i/objrow/mwl.oversampling;
    spectra[i] = exp(-pow(i*2./nlambda-1,2))/nlambda; //gaussian, -1,1 with sigma=1
    spectrafile<<lambdas[i]<<" "<<spectra[i]<<endl;
  }
  mwl.init(objrow, objcol, nlambda, lambdas, spectra);
  Real *d_patternSum;
  complexFormat *single = (complexFormat*)memMngr.borrowCache(mwl.row*mwl.column*sizeof(Real)*2);
  complexFormat *d_CpatternSum;
  Real *d_solved;
  cudaMalloc((void**)&d_patternSum, mwl.row*mwl.column*sizeof(Real));
  cudaMalloc((void**)&d_CpatternSum, mwl.row*mwl.column*sizeof(Real)*2);
  cudaMalloc((void**)&d_solved, mwl.row*mwl.column*sizeof(Real)*2);
  for(int j = 0; j < 1; j++){
    if(mwl.domnist) mnist_dat->cuRead(d_input);
    mwl.generateMWL(d_input, d_patternSum, single);
    cudaF(extendToComplex)(d_patternSum, d_CpatternSum);
    mwl.plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
    //mwl.solveMWL(d_CpatternSum, d_solved, single);
    mwl.solveMWL(d_CpatternSum, d_solved);
    mwl.plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(j)).c_str(), 0);
    mwl.propagate(d_solved, d_CpatternSum,1);
    mwl.plt.plotComplex(d_CpatternSum, MOD, 1, 1, ("autocsolved"+to_string(j)).c_str(), 0);

  }

  return 0;
}

