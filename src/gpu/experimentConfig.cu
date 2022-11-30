#include "experimentConfig.h"
#include "cudaConfig.h"

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
  if(cuCabsf(amp[index])!=0) amp[index] = cuCmulf(amp[index], p);
}

void opticalPropagate(complexFormat* field, Real lambda, Real d, Real imagesize, int rows, int cols){
  cudaF(multiplyFresnelPhase_Device)((complexFormat*)field, M_PI/lambda/d*(imagesize*imagesize/rows/cols));
  cudaF(cudaConvertFO)(field);
  myCufftExec(*plan, (complexFormat*)field, (complexFormat*)field, CUFFT_FORWARD);
  cudaF(applyNorm)((complexFormat*)field, 1./sqrt(rows*cols));
  cudaF(cudaConvertFO)(field);
  cudaF(multiplyPatternPhase_Device)((complexFormat*)field, M_PI*lambda*d/(imagesize*imagesize), 2*d*M_PI/lambda - M_PI/2);
}

void experimentConfig::createBeamStop(){
  C_circle cir;
  cir.x0=row/2;
  cir.y0=column/2;
  cir.r=beamStopSize;
  decltype(cir) *cuda_spt;
  gpuErrchk(cudaMalloc((void**)&cuda_spt,sizeof(cir)));
  cudaMemcpy(cuda_spt, &cir, sizeof(cir), cudaMemcpyHostToDevice);
  cudaF(createMask)(beamstop, cuda_spt,1);
  cudaFree(cuda_spt);
}
void experimentConfig::propagate(void* datain, void* dataout, bool isforward){
  myCufftExec( *plan, (complexFormat*)datain, (complexFormat*)dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
  applyNorm<<<numBlocks,threadsPerBlock>>>((complexFormat*)dataout, isforward? forwardFactor: inverseFactor);
}
void experimentConfig::multiplyPatternPhase(void* amp, Real distance){
  multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, pixelsize*pixelsize*M_PI/(distance*lambda), 2*distance*M_PI/lambda-M_PI/2);
}
void experimentConfig::multiplyPatternPhase_factor(void* amp, Real factor1, Real factor2){
  multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, factor1, factor2-M_PI/2);
}
void experimentConfig::multiplyFresnelPhase(void* amp, Real distance){
  Real fresfactor = M_PI*lambda*distance/(pow(pixelsize*row,2));
  multiplyFresnelPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, fresfactor);
}
void experimentConfig::multiplyFresnelPhase_factor(void* amp, Real factor){
  multiplyFresnelPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, factor);
}
void experimentConfig::calculateParameters(){
  enhancement = pow(pixelsize,2)*sqrt(row*column)/(lambda*d); // this guarentee energy conservation
  fresnelFactor = lambda*d/pow(pixelsize,2)/row/column;
  forwardFactor = fresnelFactor*enhancement;
  inverseFactor = 1./row/column/forwardFactor;
}
