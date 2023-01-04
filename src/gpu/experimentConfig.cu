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

__global__ void multiplyPatternPhaseOblique_Device(complexFormat* amp, Real r_d_lambda, Real d_r_lambda, Real costheta){ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda and costheta = z/r
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  Real phase = (pow((x-(cuda_row>>1)*costheta),2)+pow(y-(cuda_column>>1),2))*r_d_lambda+d_r_lambda;
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

__global__ void multiplyFresnelPhaseOblique_Device(complexFormat* amp, Real phaseFactor, Real costheta_r){ // costheta_r = 1./costheta = r/z
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  Real phase = phaseFactor*(pow((x-(cuda_row>>1))*costheta_r,2)+pow(y-(cuda_column>>1),2));
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
  cuda_spt = (decltype(cir)*)memMngr.borrowCache(sizeof(cir));
  cudaMemcpy(cuda_spt, &cir, sizeof(cir), cudaMemcpyHostToDevice);
  beamstop = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  cudaF(createMask)(beamstop, cuda_spt,1);
  memMngr.returnCache(cuda_spt);
}
void experimentConfig::propagate(void* datain, void* dataout, bool isforward){
  myCufftExec( *plan, (complexFormat*)datain, (complexFormat*)dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
  applyNorm<<<numBlocks,threadsPerBlock>>>((complexFormat*)dataout, isforward? forwardFactor: inverseFactor);
}
void experimentConfig::multiplyPatternPhase(void* amp, Real distance){
  if(costheta == 1){
    multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp,
         pixelsize*pixelsize*M_PI/(distance*lambda),  2*distance*M_PI/lambda-M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp,
         pixelsize*pixelsize*M_PI/(distance*lambda),  2*distance*M_PI/lambda-M_PI/2, costheta);
  }
}
void experimentConfig::multiplyPatternPhase_reverse(void* amp, Real distance){
  if(costheta == 1){
    multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp,
        -pixelsize*pixelsize*M_PI/(distance*lambda), -2*distance*M_PI/lambda+M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp,
        -pixelsize*pixelsize*M_PI/(distance*lambda), -2*distance*M_PI/lambda+M_PI/2, costheta);
  }
}
void experimentConfig::multiplyPatternPhase_factor(void* amp, Real factor1, Real factor2){
  if(costheta == 1){
    multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, factor1, factor2-M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, factor1, factor2-M_PI/2, costheta);
  }
}
void experimentConfig::multiplyFresnelPhase(void* amp, Real distance){
  Real fresfactor = M_PI*lambda*distance/(pow(pixelsize*row,2));
  if(costheta == 1){
    multiplyFresnelPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, fresfactor);
  }else{
    multiplyFresnelPhaseOblique_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, fresfactor, 1./costheta);
  }
}
void experimentConfig::multiplyFresnelPhase_factor(void* amp, Real factor){
  if(costheta == 1){
    multiplyFresnelPhase_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, factor);
  }else{
    multiplyFresnelPhaseOblique_Device<<<numBlocks,threadsPerBlock>>>((complexFormat*)amp, factor, 1./costheta);
  }
}
void experimentConfig::calculateParameters(){
  enhancement = pow(pixelsize,2)*sqrt(row*column)/(lambda*d); // this guarentee energy conservation
  fresnelFactor = lambda*d/pow(pixelsize,2)/row/column;
  forwardFactor = fresnelFactor*enhancement;
  inverseFactor = 1./row/column/forwardFactor;
}
