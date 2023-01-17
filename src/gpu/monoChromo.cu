#include "monoChromo.h"
#include "cudaConfig.h"
#include "common.h"
int nearestEven(Real x){
  return round(x/2)*2;
}
monoChromo::monoChromo(const char* configfile) : experimentConfig(configfile){}
void monoChromo::init(int nrow, int ncol, int nlambda_, Real* lambdas_, Real* spectra_){
  nlambda = nlambda_;
  spectra = spectra_;
  row = nrow*oversampling;
  column = ncol*oversampling;
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = ccmemMngr.borrowCache(sizeof(cufftHandle)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = nearestEven(row*lambdas_[i]);
    cols[i] = nearestEven(column*lambdas_[i]);
    new(&(((cufftHandle*)locplan)[i]))cufftHandle();
    cufftPlan2d ( &(((cufftHandle*)locplan)[i]), rows[i], cols[i], FFTformat);
  }
}
void monoChromo::init(int nrow, int ncol, Real* lambdasi, Real* spectrumi, Real endlambda){
  row = nrow*oversampling;
  column = ncol*oversampling;
  Real currentLambda = 1;
  int currentPoint = 0;
  int jump = 1;
  Real stepsize = 2./row*jump;
  nlambda = (endlambda-1)/stepsize;
  spectra = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
  int i = 0;
  Real tot = 0;
  while(currentLambda < endlambda){
    int count = 0;
    Real intensity = 0;
    while(lambdasi[currentPoint] < currentLambda+stepsize/2){
      count++;
      intensity += spectrumi[currentPoint];
      currentPoint++;
    }
    if(count >=2 ){ //use average
      spectra[i] = intensity/count;
    }else{ //use interpolation
      if(currentLambda == lambdasi[currentPoint-1]){
        spectra[i] = spectrumi[currentPoint-1];
      }
      else if(currentLambda > lambdasi[currentPoint-1]){
        Real dlambda = lambdasi[currentPoint]-lambdasi[currentPoint-1];
        Real dx = (currentLambda - lambdasi[currentPoint-1])/dlambda;
        spectra[i] = spectrumi[currentPoint-1]*(1-dx) + spectrumi[currentPoint]*(dx);
      }else{
        Real dlambda = lambdasi[currentPoint-1]-lambdasi[currentPoint-2];
        Real dx = (currentLambda - lambdasi[currentPoint-2])/dlambda;
        spectra[i] = spectrumi[currentPoint-2]*(1-dx) + spectrumi[currentPoint-1]*(dx);
      }
    }
    tot+=spectra[i];
    i++;
    currentLambda+=stepsize;
  }
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = ccmemMngr.borrowCache(sizeof(cufftHandle)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+i*2*jump;
    cols[i] = column+i*2*jump;
    printf("%d: (%d,%d)=%f\n",i, rows[i],cols[i],spectra[i]/=tot);
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
  if(nlambda<0) printf("nlambda not initialized: %d\n",nlambda);
  Real dt = 2;
  Real friction = 0.2;
  cudaMemcpy(d_output, initial? initial : d_input, sz, cudaMemcpyDeviceToDevice);
  init_fft(row,column);
  complexFormat *deltab = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *fbi = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *fftb = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *momentum = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *padded = (complexFormat*) memMngr.borrowCache(sizeof(complexFormat)*rows[nlambda-1]*cols[nlambda-1]);
  Real stepsize = 0.5;
  cudaMemset(momentum, 0, sz/sizeof(char));
  for(int i = 0; i < nIter; i++){
    cudaMemcpy(deltab, d_input, sz, cudaMemcpyDeviceToDevice);
    //plt.plotComplex(deltab, MOD, 0, 1, "inputmerged", 1);
    myCufftExec( *plan, (complexFormat*)d_output, fftb, CUFFT_INVERSE);
    //cudaF(cudaConvertFO)(fftb);
    //cudaF(zeroEdge)(fftb, 100);
    //cudaF(cudaConvertFO)(fftb);
    //myCufftExec( *plan, fftb, (complexFormat*)d_output, CUFFT_FORWARD);
    //cudaF(applyNorm)((complexFormat*)d_output, 1./row/column);
    cudaF(forcePositive)((complexFormat*)d_output);
    cudaF(cudaConvertFO)(fftb);
    cudaF(add)(deltab, (complexFormat*)d_output, -spectra[0]);
    for(int j = 1; j < nlambda; j++){
      if(spectra[j]<=0) continue;
      size_t N = rows[j]*cols[j];
      init_cuda_image(rows[j], cols[j], 65536, 1);
      cudaF(pad)(fftb, padded, row, column);
      cudaF(cudaConvertFO)(padded);
      cudaF(applyNorm)(padded, spectra[j]/N);
      //cudaF(applyNorm)(padded, spectra[j]/sqrt(N)/sqrt(row*column));
      //cudaF(applyNorm)(padded, spectra[j]/(row*column));
      myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_FORWARD);
      init_cuda_image(row, column, 65536, 1);
      cudaF(crop)(padded, fbi, rows[j], cols[j]);
      cudaF(add)(deltab, fbi, -1);
    }
    cudaF(add)((complexFormat*)momentum, deltab, stepsize*spectra[0]);
    //cudaF(add)((complexFormat*)d_output, deltab, stepsize*spectra[0]);
    if(i==nIter-1) plt.plotComplex(deltab, MOD, 0, 1, "residual", 1);
    for(int j = 1; j < nlambda; j++){
      if(spectra[j]<=0) continue;
      //int N = int(rows[j]) * int(cols[j]);
      init_cuda_image(rows[j], cols[j], 65536, 1);
      cudaF(pad)((complexFormat*)deltab, padded, row, column);
      myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_INVERSE);
      cudaF(cudaConvertFO)(padded);
      init_cuda_image(row, column, 65536, 1);
      cudaF(crop)(padded, fbi, rows[j], cols[j]);
      cudaF(cudaConvertFO)(fbi);
      //cudaF(applyNorm)(fbi, 1./N);
      //cudaF(applyNorm)(fbi, 1./sqrt(N)/sqrt(row*column));
      cudaF(applyNorm)(fbi, 1./(row*column));
      myCufftExec( *plan, fbi, fbi, CUFFT_FORWARD);
      //plt.plotComplex(fbi, MOD, 0, 1, ("fbi"+to_string(j)).c_str(), 1);
      cudaF(add)((complexFormat*)momentum, fbi, stepsize*spectra[j]);
      //cudaF(add)((complexFormat*)d_output, fbi, stepsize*spectra[j]);
    }
    cudaF(applyNorm)((complexFormat*)momentum, 1-friction*dt);
    cudaF(add)((complexFormat*)d_output, momentum, dt);
  }
  memMngr.returnCache(padded);
  memMngr.returnCache(fbi);
  memMngr.returnCache(fftb);
  memMngr.returnCache(deltab);
}
