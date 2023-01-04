#include "monoChromo.h"
#include "cudaConfig.h"
#include "common.h"
int nearestEven(Real x){
  return round(x/2)*2;
}
monoChromo::monoChromo(const char* configfile, int nlambda_, Real* lambdas_, Real* spectra_) : experimentConfig(configfile), nlambda(nlambda_),
  spectra(spectra_), lambdas(lambdas_){}
void monoChromo::init(int nrow, int ncol){
  row = nrow*oversampling;
  column = ncol*oversampling;
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = ccmemMngr.borrowCache(sizeof(cufftHandle)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = nearestEven(row*lambdas[i]);
    cols[i] = nearestEven(column*lambdas[i]);
    new(&(((cufftHandle*)locplan)[i]))cufftHandle();
    cufftPlan2d ( &(((cufftHandle*)locplan)[i]), rows[i], cols[i], FFTformat);
  }
}
void monoChromo::generateMWL(void* d_input, void* d_patternSum, void* single){
  Real *d_pattern = (Real*) memMngr.borrowCache(row*column*sizeof(Real));
  complexFormat *d_intensity = (complexFormat*)memMngr.borrowCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat* d_patternAmp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(Real)*2);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    init_cuda_image(thisrow, thiscol, 65536, 1);
    cudaF(createWaveFront)((Real*)d_input, 0, (complexFormat*)d_intensity, row/oversampling, column/oversampling);
    myCufftExec( ((cufftHandle*)locplan)[i], d_intensity,d_intensity, CUFFT_FORWARD);
    cudaF(cudaConvertFO)(d_intensity);
    init_cuda_image(row, column, 65536, 1);
    cudaF(crop)(d_intensity,d_patternAmp,thisrow,thiscol);
    cudaF(applyNorm)(d_patternAmp, sqrt(exposure*spectra[i])/sqrt(thiscol*thisrow));
    if(i==0) {
      cudaF(getMod2)((Real*)d_patternSum, d_patternAmp);
      if(single!=0) {
        cudaF(extendToComplex)((Real*)d_patternSum, (complexFormat*)single);
        cudaF(applyNorm)((complexFormat*)single, 1./spectra[i]);
      }
    }else{
      cudaF(getMod2)(d_pattern, d_patternAmp);
      cudaF(add)((Real*)d_patternSum, (Real*)d_pattern, 1);
    }
  }
  memMngr.returnCache(d_pattern);
  memMngr.returnCache(d_intensity);
  memMngr.returnCache(d_patternAmp);
}
void monoChromo::solveMWL(void* d_input, void* d_output, void* initial)
{
  int sz = row*column*sizeof(complexFormat);
  cudaMemcpy(d_output, initial? initial : d_input, sz, cudaMemcpyDeviceToDevice);
  init_fft(row,column);
  complexFormat *deltab = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *fbi = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *fftb = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *padded = (complexFormat*) memMngr.borrowCache(sizeof(complexFormat)*rows[nlambda-1]*cols[nlambda-1]);
  Real stepsize = 1;
  for(int i = 0; i < nIter; i++){
    cudaMemcpy(deltab, d_input, sz, cudaMemcpyDeviceToDevice);
    //plt.plotComplex(deltab, MOD, 0, 1, "inputmerged", 1);
    cudaF(add)(deltab, (complexFormat*)d_output, -spectra[0]);
    myCufftExec( *plan, (complexFormat*)d_output, fftb, CUFFT_INVERSE);
    cudaF(cudaConvertFO)(fftb);
    for(int j = 1; j < nlambda; j++){
      size_t N = rows[j]*cols[j];
      init_cuda_image(rows[j], cols[j], 65536, 1);
      cudaF(pad)(fftb, padded, row, column);
      cudaF(cudaConvertFO)(padded);
      //cudaF(applyNorm)(padded, spectra[j]/sqrt(N)/sqrt(row*column));
      cudaF(applyNorm)(padded, spectra[j]/N);
      myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_FORWARD);
      init_cuda_image(row, column, 65536, 1);
      cudaF(crop)(padded, fbi, rows[j], cols[j]);
      cudaF(add)(deltab, fbi, -1);
    }
    cudaF(add)((complexFormat*)d_output, deltab, stepsize*spectra[0]);
    if(i==nIter-1) plt.plotComplex(deltab, MOD, 0, 1, "residual", 1);
    for(int j = 1; j < nlambda; j++){
      int N = int(rows[j]) * int(cols[j]);
      init_cuda_image(rows[j], cols[j], 65536, 1);
      cudaF(pad)((complexFormat*)deltab, padded, row, column);
      myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_INVERSE);
      cudaF(cudaConvertFO)(padded);
      init_cuda_image(row, column, 65536, 1);
      cudaF(crop)(padded, fbi, rows[j], cols[j]);
      cudaF(cudaConvertFO)(fbi);
      //cudaF(applyNorm)(fbi, 1./sqrt(N)/sqrt(row*column));
      cudaF(applyNorm)(fbi, 1./N);
      myCufftExec( *plan, fbi, fbi, CUFFT_FORWARD);
      //plt.plotComplex(fbi, MOD, 0, 1, ("fbi"+to_string(j)).c_str(), 1);
      cudaF(add)((complexFormat*)d_output, fbi, stepsize*spectra[j]);
    }
    cudaF(forcePositive)((complexFormat*)d_output);
  }
  memMngr.returnCache(padded);
  memMngr.returnCache(fbi);
  memMngr.returnCache(fftb);
  memMngr.returnCache(deltab);
}
