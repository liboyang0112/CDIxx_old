#include <complex>
# include <cassert>
# include <stdio.h>
# include <time.h>
# include <random>

#include <stdio.h>
#include "fftw.h"
#include <iostream>
#include <fstream>
#include <libconfig.h++>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "cufft.h"

#include "common.h"
#include "imageReader.h"
#include <ctime>
#include "cudaConfig.h"
#include "readConfig.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

cufftHandle *plan = new cufftHandle();
static dim3 numBlocks;
//#define Bits 16
using namespace cv;
Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*pi*sigma*sigma)*gaussian(x,y,sigma);
}

class ImageMask{
public:
  int nrow;
  int ncol;
  size_t sz;
  Real *data;
  Mat *image;
  Real threshold;
  ImageMask *cuda;
  ImageMask(){
    cudaMalloc(&cuda, sizeof(ImageMask));
  };
  void init_image(Mat* image_){
    nrow = image_->rows;
    ncol = image_->cols;
    image = image_;
    sz = image_->total()*sizeof(Real);
    cudaMalloc((void**)&data,sz);
  }
  void updateCuda(){
    cudaMemcpy(cuda, this, sizeof(ImageMask), cudaMemcpyHostToDevice);
  }
  void cpyToGM(){
    cudaMemcpy(data, image->data, sz, cudaMemcpyHostToDevice);
  }
  void cpyFromGM(){
    cudaMemcpy(image->data, data, sz, cudaMemcpyDeviceToHost);
  }
  __device__ __host__ bool isInside(int x, int y){
    if(data[x*ncol+y] < threshold) {
	    //printf("%d, %d = %f lower than threshold, dropping\n",x,y,image->ptr<Real>(x)[y]);
	    return false;
    }
    return true;
  }
};
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

Mat* gaussianKernel(int rows, int cols, Real sigma){
  Mat* image = new Mat(rows, cols, float_cv_format(1));
  auto f = [&](int x, int y, Real &data){
    data = gaussian_norm(x-rows/2,y-cols/2,sigma);
  };
  imageLoop<decltype(f), Real>(image,&f);
  
  return image;
}
__global__ void applyNorm(complexFormat* data, double factor){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  data[index].x*=factor;
  data[index].y*=factor;
}

__global__ void applyMod(complexFormat* source, complexFormat* target, ImageMask *bs = 0, bool loose=0, int iter = 0){
  assert(source!=0);
  assert(target!=0);
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  Real maximum = pow(mergeDepth,2)*cuda_scale*0.99;
  int index = x*cuda_column + y;
  complexFormat targetdata = target[index];
  Real mod2 = targetdata.x*targetdata.x + targetdata.y*targetdata.y;
  if(loose && bs && bs->isInside(x,y)) {
	  if(iter > 500) return;
	  else mod2 = maximum+1;
  }
  Real tolerance = 1./cuda_rcolor*cuda_scale;//+30./cuda_rcolor; // fluctuation caused by bit depth and noise
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
__global__ void multiplyPatternPhase_Device(complexFormat* amp, Real r_d_lambda, Real d_r_lambda){ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  Real phase = (pow(x-(cuda_row>>1),2)+pow(y-(cuda_column>>1),2))*r_d_lambda+d_r_lambda;
  int index = x*cuda_column + y;
  complexFormat p = {cos(phase),sin(phase)};
  p = cuCmulf(amp[index], p);
  amp[index].x = p.x;
  amp[index].y = p.y;
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
  Real enhancement = 0;
  Real forwardFactor = 0;
  Real fresnelFactor = 0;
  Real inverseFactor = 0;
  ImageMask* spt=0;
  ImageMask* beamStop=0;

  void propagate(complexFormat* datain, complexFormat* dataout, bool isforward){
      myCufftExec( *plan, datain, dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
      applyNorm<<<numBlocks,threadsPerBlock>>>(dataout, isforward? forwardFactor: inverseFactor);
  }
  void multiplyPatternPhase(complexFormat* amp){
    multiplyPatternPhase_Device<<<numBlocks,threadsPerBlock>>>(amp, pixelsize*pixelsize*M_PI/(d*lambda), 2*d*M_PI/lambda);
  }
  void init(int row, int column){
    enhancement = pow(pixelsize,4)*row*column/pow(lambda*d,2); // this guarentee energy conservation
    enhancement *= exposure; //too strong in the middle
    fresnelFactor = lambda*d/pow(pixelsize,2)/row/column;
    forwardFactor = fresnelFactor*enhancement;
    inverseFactor = 1./row/column/forwardFactor;
  }
};

__global__ void applySupport(complexFormat *gkp1, complexFormat *gkprime, Real* objMod, ImageMask *spt, Real fresnelFactor = 0){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = x*cuda_column + y;
  if(x >= cuda_row || y >= cuda_column) return;

  //epsilonF+=hypot(gkp1data[0]-gkprimedata[0],gkp1data[1]-gkprimedata[1]);
  //fftw_format tmp = {gkp1data[0],gkp1data[1]};
  bool inside = spt->isInside(x,y);
  complexFormat &gkp1data = gkp1[index];
  complexFormat &gkprimedata = gkprime[index];
  //if(iter >= niters - 20 ) ApplyERSupport(inside,gkp1data,gkprimedata);
  //if(iter >= niters - 20 || iter % 200 == 0) ApplyERSupport(inside,gkp1data,gkprimedata);
  //if(iter >= niters - 20 || iter<20) ApplyERSupport(inside,gkp1data,gkprimedata);
  //if(iter >= niters - 20) ApplyERSupport(inside,gkp1data,gkprimedata);
  //ApplyERSupport(inside,gkp1data,gkprimedata);
  //else ApplyHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
  //else ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
  //printf("%d, (%f,%f), (%f,%f), %f\n",inside, gkprimedata.x,gkprimedata.y,gkp1data.x,gkp1data.y,cuda_beta_HIO);
  ApplyHIOSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  if(fresnelFactor*(cuda_row*cuda_row+cuda_column*cuda_column)>1) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(pow(x-(cuda_row>>1),2)+pow(y-(cuda_column>>1),2));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
  objMod[index] = cuCabsf(gkp1data);
  //Real thres = gaussian(x-row/2,y-column/2,40);
  //ApplyLoosePOSHIOSupport(inside,gkp1data,gkprimedata,beta_HIO,thres);
  //ApplyLoosePOSERSupport(inside,gkp1data,gkprimedata,thres);
  //else {
  //ApplyDMSupport(inside,gkp1data, gkprimedata, pmpsg[index], gammas, gammam, beta);
  //}
  //ApplyERSupport(inside,pmpsg[index],gkp1data);
  //ApplyHIOSupport(inside,gkp1data,gkprimedata,beta);
  //else ApplySFSupport(inside,gkp1data,gkprimedata);
  //epsilonS+=hypot(tmp[0]-gkp1data[0],tmp[1]-gkp1data[1]);
}
Mat* phaseRetrieve( experimentConfig &setups, Mat* targetfft, Mat* &gkp1, Mat *cache = 0, Mat* fftresult = 0 ){
    Mat* pmpsg = 0;
    bool useShrinkMap = setups.useShrinkMap;
    int row = targetfft->rows;
    int column = targetfft->cols;
    bool useDM = setups.useDM;
    bool useBS = setups.useBS;
    ImageMask &re = *setups.spt;
    auto &cir = *(setups.beamStop);
    if(useDM) {
      pmpsg = new Mat();
      fftresult->copyTo(*pmpsg);
    }
    if(gkp1==0) gkp1 = new Mat(row,column,float_cv_format(2));
    assert(targetfft!=0);
    Real beta = -1;
    Real beta_HIO = 0.9;
    cudaMemcpyToSymbol(cuda_beta_HIO,&beta_HIO,sizeof(beta_HIO));
    Real gammas = -1./beta;
    Real gammam = 1./beta;
    Real epsilonS, epsilonF;
    std::ofstream fepF,fepS;
    fepF.open("epsilonF.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    fepS.open("epsilonS.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    int niters = 5000;
    int tot = row*column;
    Mat objMod(row,column,float_cv_format(1));
    Mat* maskKernel;
    Real gaussianSigma = 3;

    size_t sz = row*column*sizeof(complexFormat);
    complexFormat *cuda_fftresult, *cuda_targetfft, *cuda_gkprime, *cuda_gkp1, *cuda_pmpsg;
    Real *cuda_objMod;
    ImageMask *cuda_spt;
    gpuErrchk(cudaMalloc((void**)&cuda_fftresult, sz));
    cudaMalloc((void**)&cuda_targetfft, sz);
    cudaMalloc((void**)&cuda_gkprime, sz);
    cudaMalloc((void**)&cuda_gkp1, sz);
    cudaMalloc((void**)&cuda_objMod, sz/2);
    cudaMalloc((void**)&cuda_spt, sizeof(ImageMask));
    cudaMemcpy(cuda_spt, &re, sizeof(ImageMask), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_targetfft, targetfft->data, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_fftresult, fftresult->data, sz, cudaMemcpyHostToDevice);

    cudaMemcpy(gkp1->data, cuda_targetfft, sz, cudaMemcpyDeviceToHost);
    convertFromComplexToInteger( gkp1,cache, MOD2,1);
    imwrite("pre_recon_intensity.png",*cache);

    setups.propagate(cuda_targetfft, cuda_gkp1,0);

    std::chrono::time_point<std::chrono::high_resolution_clock> now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<int64_t, std::nano> time_applyMod(0);
    std::chrono::duration<int64_t, std::nano> time_FFT(0);
    std::chrono::duration<int64_t, std::nano> time_support(0);
    std::chrono::duration<int64_t, std::nano> time_norm(0);
    for(int iter = 0; iter < niters; iter++){
      //start iteration
      if(iter%100==0) {
	long tot = time_FFT.count()+time_norm.count()+time_support.count()+time_applyMod.count();
	if(tot!=0)
          printf("iter: %d, timing:\n  FFT:%ld, %4.2f%%\n  NORM:%ld, %4.2f%%\n  Support:%ld, %4.2f%%\n  applyMod:%ld, %4.2f%%\n",iter, 
	  time_FFT.count(),     ((Real)time_FFT.count())/tot*100,
	  time_norm.count(),    ((Real)time_norm.count())/tot*100,
	  time_support.count(), ((Real)time_support.count())/tot*100,
	  time_applyMod.count(),((Real)time_applyMod.count())/tot*100
        );
	if(setups.saveIter){
          cudaMemcpy(gkp1->data, cuda_gkp1, sz, cudaMemcpyDeviceToHost);
          convertFromComplexToInteger( gkp1,cache, MOD2,0);
          std::string iterstr = to_string(iter);
          imwrite("recon_intensity"+iterstr+".png",*cache);
          convertFromComplexToInteger( gkp1,cache, PHASE,0);
          imwrite("recon_phase"+iterstr+".png",*cache);
	}
      }
      now = std::chrono::high_resolution_clock::now();
      if(useBS) applyMod<<<numBlocks,threadsPerBlock>>>(cuda_fftresult,cuda_targetfft,cir.cuda, !setups.reconAC || iter > 1000,iter);  //apply mod to fftresult, Pm
      else applyMod<<<numBlocks,threadsPerBlock>>>(cuda_fftresult,cuda_targetfft,0, !setups.reconAC || iter > 1000,iter);  //apply mod to fftresult, Pm
      if(useDM) {
        if(useBS) applyMod<<<numBlocks,threadsPerBlock>>>(cuda_pmpsg,cuda_targetfft,cir.cuda, !setups.reconAC || iter > 1000,iter);  
        else applyMod<<<numBlocks,threadsPerBlock>>>(cuda_pmpsg,cuda_targetfft,0, !setups.reconAC || iter > 1000, iter);
      }
      time_applyMod+=std::chrono::high_resolution_clock::now()-now;
      
      epsilonS = epsilonF = 0;
      now = std::chrono::high_resolution_clock::now();
      setups.propagate(cuda_fftresult, cuda_gkprime, 0);
      time_FFT+=std::chrono::high_resolution_clock::now()-now;
      if(useDM){
        now = std::chrono::high_resolution_clock::now();
	setups.propagate(cuda_pmpsg, cuda_pmpsg, 0);
        time_FFT+=std::chrono::high_resolution_clock::now()-now;
      }
      bool updateMask = (iter > 1000) && iter%20==0 && useShrinkMap && iter!=0;
      if(updateMask){
        int size = floor(gaussianSigma*6); // r=3 sigma to ensure the contribution is negligible (0.01 of the maximum)
        size = size/2*2+1; //ensure odd
        maskKernel = gaussianKernel(size,size,gaussianSigma);
      }
      now = std::chrono::high_resolution_clock::now();
      applySupport<<<numBlocks,threadsPerBlock>>>(cuda_gkp1, cuda_gkprime, cuda_objMod,cuda_spt,iter>1000?0:setups.fresnelFactor);
      time_support+=std::chrono::high_resolution_clock::now()-now;

      //if(iter==1){
      //    cudaMemcpy(gkp1->data, cuda_gkp1, sz, cudaMemcpyDeviceToHost);
      //    convertFromComplexToInteger( gkp1,cache, MOD2,1);
      //    imwrite("debug1.png",*cache);
      //exit(0);
      //}

      if(updateMask){
        cudaMemcpy(objMod.data, cuda_objMod, sz/2, cudaMemcpyDeviceToHost);
        filter2D(objMod, *re.image,objMod.depth(),*maskKernel);
	((ImageMask*)&re)->cpyToGM();
	if(gaussianSigma>1.5) gaussianSigma*=0.99;
	delete maskKernel;
      }
      if(updateMask&&iter%100==0&&setups.saveIter){
	convertFromComplexToInteger<Real>(re.image, cache,MOD,0);
        std::string iterstr = to_string(iter);
	imwrite("mask"+iterstr+".png",*cache);
      }
      if(iter!=0){
        fepF<<sqrt(epsilonF/tot)<<endl;
        fepS<<sqrt(epsilonS/tot)<<endl;
      }else {
        cudaMemcpy(gkp1->data, gkp1, sz, cudaMemcpyDeviceToHost);
        convertFromComplexToInteger( gkp1,cache, MOD2,0);
        imwrite("recon_support.png",*cache);
        convertFromComplexToInteger( gkp1,cache, PHASE,0);
        imwrite("recon_phase_support.png",*cache);
      }

      //if(sqrt(epsilonS/row/column)<0.05) break;
      now = std::chrono::high_resolution_clock::now();
      setups.propagate( cuda_gkp1, cuda_fftresult, 1);
      time_FFT+=std::chrono::high_resolution_clock::now()-now;
      if(useDM){ // FFT to get f field;
        setups.propagate(cuda_pmpsg, cuda_pmpsg, 1);
        cudaDeviceSynchronize();
      }
      //end iteration
    }
    fepF.close();
    fepS.close();
    cudaMemcpy(fftresult->data, cuda_fftresult, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(targetfft->data, cuda_targetfft, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(gkp1->data, cuda_gkp1, sz, cudaMemcpyDeviceToHost);

    convertFromComplexToInteger( gkp1,cache, MOD2,0);
    imwrite("recon_intensity.png",*cache);
    convertFromComplexToInteger(gkp1, cache, PHASE,0);
    imwrite("recon_phase.png",*cache);
    if(useDM)  convertFromComplexToInteger( pmpsg, cache, MOD2,1);
    if(useDM)  imwrite("recon_pmpsg.png",*cache);
    convertFromComplexToInteger( fftresult, cache, MOD2,1);
    imwrite("recon_pattern.png",*cache);
    return fftresult;
}

__global__ void applyAutoCorrelationMod(complexFormat* source,complexFormat* target, ImageMask *bs = 0){
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
  if(bs && bs->isInside(x,y)) {
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

template <typename sptType>
void autoCorrelationConstrain(Mat* pattern, sptType *spt, Mat* cache, Real norm, ImageMask *bs = 0){  //beam stop
  Real totalIntensity = 1000;
  cudaMemcpyToSymbol(cuda_totalIntensity,&totalIntensity,sizeof(totalIntensity));
  complexFormat *autocorrelation, *cuda_pattern, *autoprime;
  ImageMask autoCorrelationMask;
  ImageMask *cuda_Mask;
  autoCorrelationMask.init_image(new Mat(pattern->rows,pattern->cols,float_cv_format(1)));
  autoCorrelationMask.threshold = 0.008;
  sptType *cuda_spt;
  size_t sz = pattern->total()*sizeof(complexFormat);
  Real *cuda_objMod;
  cudaMalloc((void**)&cuda_objMod, sz/2);
  cudaMalloc((void**)&autocorrelation,sz);
  cudaMalloc((void**)&autoprime,sz);
  cudaMalloc((void**)&cuda_pattern,sz);
  cudaMalloc((void**)&cuda_spt,sizeof(sptType));
  cudaMalloc((void**)&cuda_Mask,sizeof(ImageMask));
  cudaMemcpy(cuda_pattern, pattern->data, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_spt, spt, sizeof(sptType), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_Mask, &autoCorrelationMask, sizeof(ImageMask), cudaMemcpyHostToDevice);
  createMask<sptType><<<numBlocks,threadsPerBlock>>>(autoCorrelationMask.data, cuda_spt,1);

  autoCorrelationMask.cpyFromGM();
  convertFromComplexToInteger<Real>(autoCorrelationMask.image, cache,MOD, 1);
  imwrite("autoCorrelationMask.png",*cache);
  myCufftExec(*plan,cuda_pattern,autocorrelation,CUFFT_INVERSE);
  applyNorm<<<numBlocks,threadsPerBlock>>>(autocorrelation,1./pattern->rows/pattern->cols/norm);
  cudaMemcpy(pattern->data, autocorrelation, sz, cudaMemcpyDeviceToHost);
  convertFromComplexToInteger(pattern, cache, REAL, 1);
  imwrite("initAC.png",*cache);
  Real gaussianSigma=3;
  Mat* maskKernel;
  for(int iter = 0; iter < 5000; iter++){
    bool updateMask = iter%20==0 && iter>2000;
    if(iter%100==0){
      printf("auto correlation iteration: %d\n",iter);
      cudaMemcpy(pattern->data, autoprime, sz, cudaMemcpyDeviceToHost);
      convertFromComplexToInteger(pattern, cache, REAL, 1);
      imwrite("accorrected"+to_string(iter)+".png",*cache);
    }
    if(iter<4500) applyHIOACSupport<<<numBlocks,threadsPerBlock>>>(autocorrelation, autoprime, cuda_Mask, cuda_objMod);
    else applyERACSupport<<<numBlocks,threadsPerBlock>>>(autocorrelation, autoprime, cuda_Mask, cuda_objMod);
    myCufftExec(*plan,autocorrelation,autoprime,CUFFT_FORWARD);
    applyNorm<<<numBlocks,threadsPerBlock>>>(autoprime,norm);
    applyAutoCorrelationMod<<<numBlocks,threadsPerBlock>>>(autoprime, cuda_pattern, bs?bs->cuda:0);
    myCufftExec(*plan,autoprime,autoprime,CUFFT_INVERSE);
    applyNorm<<<numBlocks,threadsPerBlock>>>(autoprime,1./pattern->rows/pattern->cols/norm);
    if(updateMask){
      int size = floor(gaussianSigma*6); // r=3 sigma to ensure the contribution is negligible (0.01 of the maximum)
      size = size/2*2+1; //ensure odd
      maskKernel = gaussianKernel(size,size,gaussianSigma);
      cudaMemcpy(autoCorrelationMask.image->data, cuda_objMod, sz/2, cudaMemcpyDeviceToHost);
      filter2D(*autoCorrelationMask.image, *autoCorrelationMask.image,autoCorrelationMask.image->depth(),*maskKernel);
      autoCorrelationMask.cpyToGM();
      if(gaussianSigma>1.5) gaussianSigma*=0.99;
      delete maskKernel;
    }
  }
  cudaFree(cuda_pattern);
  cudaFree(cuda_spt);
  cudaFree(cuda_Mask);
  cudaFree(autocorrelation);
  cudaMemcpy(pattern->data, autoprime, sz, cudaMemcpyDeviceToHost);
  cudaFree(autoprime);
  convertFromComplexToInteger(pattern, cache, REAL, 1);
  imwrite("accorrected.png",*cache);
}

int main(int argc, char** argv )
{
    experimentConfig setups(argv[1]);
    if(argc < 2){
      printf("please feed the object intensity and phase image\n");
    }
    auto seed = (unsigned)time(NULL);
    printf("command:");

    for(int i = 0; i < argc ; i++){
	    printf("%s ",argv[i]);
    }
    printf("\n");

    srand(seed);
    printf("seed:%d\n",seed);
    Real padAutoCorrelation = 0;
    Mat* gkp1 = 0;
    Mat* targetfft = 0;
    Mat* fftresult = 0;
    int row, column;
    setups.print();
    Mat intensity = readImage( setups.runSim?setups.common.Intensity.c_str():setups.common.Pattern.c_str() , !setups.runSim);
    //Mat ele = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));
    //filter2D(intensity, intensity, intensity.depth(), ele);
    //erode( intensity, intensity, ele);
    //dilate( intensity, intensity, ele);
    row = intensity.rows;
    column = intensity.cols;

    pixeltype *rowp;
    if(setups.useRectHERALDO){
      for(int i = 0; i < row ; i++){
        rowp = intensity.ptr<pixeltype>(i);
        for(int j = 0; j < column ; j++){
          if(i > row/2 || j > column/2) rowp[j] = rcolor-1;
	}
      }
    }

    if(setups.runSim){
	    row*=setups.oversampling;
	    column*=setups.oversampling;
    }

    //--------------------------some mask shapes-----------------------------------
    C_circle cir,cir2,cir3;
    //cir is the beam stop
    //cir.x0=row/2-50;
    //cir.y0=column/2+20;
    //cir.r=50;
    cir.x0=row/2;
    cir.y0=column/2;
    cir.r=15;
    //cir2.x0 = column*2/3-50;
    //cir2.y0 = row*2/3+110;
    //cir2.r = 50;
    cir2.x0 = column/2;
    cir2.y0 = row/2;
    cir2.r = 40;
    cir3.x0 = row/2;
    cir3.y0 = column/2;
    //cir3.r = 300/mergeDepth;
    cir3.r = 40;
    rect re;
    re.startx = (setups.oversampling-1)/2*row/setups.oversampling-1;
    re.starty = (setups.oversampling-1)/2*column/setups.oversampling-1;
    //re.startx = 1./4*row;
    //re.starty = 1./4*column;
    re.endx = row-re.startx;
    re.endy = column-re.starty;
    
    //------------------------configure cuda-----------------------------------------
    cudaMemcpyToSymbol(cuda_row,&row,sizeof(row));
    cudaMemcpyToSymbol(cuda_column,&column,sizeof(column));
    numBlocks.x=(row-1)/threadsPerBlock.x+1;
    numBlocks.y=(column-1)/threadsPerBlock.y+1;

    //------------------------create shrinking mask----------------------------------
    ImageMask shrinkingMask;
    shrinkingMask.threshold = 0.1;

    //------------------------configure beam stop------------------------------------
    ImageMask beamStop;
    beamStop.threshold = 0.5;
    beamStop.init_image(new Mat(row,column,float_cv_format(1)));
    C_circle *cuda_spt;
    cudaMalloc((void**)&cuda_spt,sizeof(C_circle));
    cudaMemcpy(cuda_spt, &cir, sizeof(C_circle), cudaMemcpyHostToDevice);
    createMask<<<numBlocks,threadsPerBlock>>>(beamStop.data, cuda_spt,1);
    beamStop.cpyFromGM();
    cudaFree(cuda_spt);
    beamStop.updateCuda();

    //-----------------------configure experiment setups-----------------------------
    setups.spt = &shrinkingMask;
    //setups.spt = &re;
    //setups.spt = &cir3;
    setups.beamStop = &beamStop;//&cir;
    setups.d = setups.oversampling*setups.pixelsize*setups.beamspotsize/setups.lambda; //distance to guarentee setups.oversampling
    Real k = pow(setups.pixelsize,2)*row/setups.lambda/setups.d; //calculate the proper distance for KCDI
    //setupsKCDI.d = k*setups.d/(k-1);
    //setups.pixelsize = 7;//setups.d/setups.oversampling/setups.beamspotsize*setups.lambda;
    setups.init(row,column);
    //setupsKCDI.init(row,column);
    printf("Imaging distance = %f\n", setups.d);
    printf("Pixel size = %f\n", setups.pixelsize);
    printf("forward norm = %f\n", setups.forwardFactor);
    printf("backward norm = %f\n", setups.inverseFactor);
    printf("fresnel factor = %f\n", setups.fresnelFactor);
    printf("enhancement = %f\n", setups.enhancement);

    bool isFarField = 0;
    Real fresnelNumber = pi*pow(setups.beamspotsize,2)/(setups.d*setups.lambda);
    if(fresnelNumber < 0.01) isFarField = 1;
    printf("Fresnel Number = %f\n",fresnelNumber);
    //these are for simulation
    Mat* cache = 0;
    Mat* cache1;

    cache1 = &intensity;
    if(setups.runSim){
      if(setups.phaseModulation){
        Mat phase = readImage(setups.common.Phase.c_str());
        if(setups.oversampling>1) {
          cache = extend(*cache1,setups.oversampling);
	  Mat* tmp = extend(phase,setups.oversampling);
          gkp1 = convertFromIntegerToComplex(*cache, *tmp ,gkp1);
	  delete tmp;
	}
        else gkp1 = convertFromIntegerToComplex(intensity,phase,gkp1);
      }else{
        if(setups.oversampling>1) cache = extend(*cache1,setups.oversampling);
	else cache = cache1;
	gkp1 = convertFromIntegerToComplex(*cache, gkp1,0,"waveFront");
      }
      if(!isFarField && setups.isFresnel){
        auto f = [&](int x, int y, fftw_format &data){
          auto tmp = (complex<Real>*)&data;
	  Real phase = M_PI*setups.fresnelFactor*(pow(x-(row>>1),2)+pow(y-(column>>1),2));
	  *tmp *= exp(complex<Real>(0,phase));
	};
        imageLoop<decltype(f)>(gkp1,&f,0);
      }
      if(setups.useGaussionLumination){
        //setups.spt = &re;
        //if(!setups.useShrinkMap) setups.spt = &cir3;
        //diffraction image, either from simulation or from experiments.
        auto f = [&](int x, int y, fftw_format &data){
          auto tmp = (complex<Real>*)&data;
          bool inside = cir3.isInside(x,y);
	  if(!inside) *tmp = 0.;
	  *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir3.r);
	  //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
	};
        imageLoop<decltype(f)>(gkp1,&f,0);
      }
      if(setups.useGaussionHERALDO){
        auto f = [&](int x, int y, fftw_format &data){
          auto tmp = (complex<Real>*)&data;
	  if(cir2.isInside(x,y)) 
            *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir2.r*4);
	  else *tmp = gaussian(x-cir2.x0,y-cir2.y0,cir2.r*4);
	  if(x < row*1/3 && y < row*1/3) *tmp = 0;
	  //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
	};
        imageLoop<decltype(f)>(gkp1,&f,0);
      }
      convertFromComplexToInteger(gkp1, cache, MOD2,0,1,"Object MOD2");
      imwrite("init_object.png",*cache);
      convertFromComplexToInteger(gkp1, cache, PHASE,0,1,"Object Phase");
      imwrite("init_object_phase.png",*cache);
      targetfft = fftw(gkp1,targetfft,1,setups.forwardFactor); 
    }else{
      //if(mergeDepth == 1) cache = cache1;
      //else 
      cache = new Mat(row, column, format_cv);
      if(cache1->depth() == CV_32F || cache1->depth() == CV_64F) 
        targetfft = cache1;
      else
        targetfft = convertFromIntegerToComplex(*cache1,targetfft,1); 
    }
    if(setups.restart){
      gkp1 = new Mat(readImage(setups.common.restart.c_str()));
      fftresult = fftw(gkp1,fftresult,1,setups.forwardFactor); //If not setups.restart, this line just allocate space, the values are not used.
    }
    //cir2.x0=row/2;
    //cir2.y0=column/2;
    Real decay = scale;
    if(setups.runSim) decay=1;
    std::default_random_engine generator;
    Real noiseLevel = 0;
    std::poisson_distribution<int> distribution(noiseLevel);
    Mat *autocorrelation = new Mat(row,column,float_cv_format(2),Scalar::all(0.));
    for(int i = 0; i<row*column; i++){ //remove the phase information
     // Real randphase = arg(tmp);//static_cast<Real>(rand())/RAND_MAX*2*pi;
      complex<Real> &data = ((complex<Real>*)(targetfft->data))[i];
      fftw_format &datacor = ((fftw_format*)autocorrelation->data)[i];
      Real mod = abs(data)*sqrt(decay);
      if(setups.runSim&&setups.simCCDbit) {
        int range= pow(2,16);
        mod = sqrt(((Real)floor(pow(mod,2)*range))/(range)); //assuming we use 16bit CCD
        mod = sqrt(max(0.,pow(mod,2)+Real(distribution(generator)-noiseLevel)/range)); //Poisson noise
        data = complex<Real>(mod,0); 
      }
      //datacor[0] = pow(mod,2)*(tx-row/2)*(ty-column/2)/90; // ucore is the derivitaves of the diffraction pattern: append *(tx-row/2)*(ty-column/2)/20;
      datacor = fftw_format(pow(mod,2),0); //ucore is the diffraction pattern
      if(padAutoCorrelation) datacor/=padAutoCorrelation;
    }
    if(!setups.reconAC || setups.runSim) {
      autocorrelation = fftw(autocorrelation, autocorrelation, 0,setups.inverseFactor);
    }
    if(padAutoCorrelation) {
      row*=padAutoCorrelation;
      column*=padAutoCorrelation;
      cudaMemcpyToSymbol(cuda_row,&row,sizeof(row));
      cudaMemcpyToSymbol(cuda_column,&column,sizeof(column));
      Mat* tmp = convertFO<complex<Real>>(autocorrelation);
      delete autocorrelation;
      autocorrelation = extend(*tmp,padAutoCorrelation);
      delete cache,targetfft,gkp1;
      cache = new Mat(row,column,format_cv);
      targetfft = new Mat(row,column,float_cv_format(2));
      gkp1 = 0;
      delete tmp;
    }
    cudaMemcpyToSymbol(cuda_rcolor,&rcolor,sizeof(rcolor));
    //Real tmp = scale*10;
    cudaMemcpyToSymbol(cuda_scale,&scale,sizeof(scale));
    cufftPlan2d ( plan, row, column, FFTformat);

    shrinkingMask.init_image(new Mat(row,column,float_cv_format(1)));
    convertFromComplexToInteger( autocorrelation, cache, REAL,1,1,"HERALDO U core"); 
    imwrite("ucore.png",*cache);
    if(!setups.reconAC || setups.runSim) {
      for(int i = 0; i<row*column; i++){ //remove the phase information
        complex<Real> &datacor = ((complex<Real>*)autocorrelation->data)[i];
	datacor = abs(datacor);
      }
      rect *cuda_spt;
      cudaMalloc((void**)&cuda_spt,sizeof(rect));
      cudaMemcpy(cuda_spt, &re, sizeof(rect), cudaMemcpyHostToDevice);
      createMask<<<numBlocks,threadsPerBlock>>>(shrinkingMask.data, cuda_spt,0);
      shrinkingMask.cpyFromGM();
      cudaFree(cuda_spt);
      shrinkingMask.updateCuda();
    }
    else {
      autoCorrelationConstrain(autocorrelation, &re, cache, setups.forwardFactor, setups.useBS?&beamStop:0);
      fftw(autocorrelation, targetfft, 1, setups.forwardFactor);
      auto f =  [&](int x, int y, fftw_format &data){
	data = sqrt(max(data.real(),0.));
      };
      imageLoop<decltype(f)>(targetfft, &f, 0);
      auto f1 = [&](int x, int y, Real &data, fftw_format &dataout){
        data = abs(dataout)>shrinkingMask.threshold;
      };
      imageLoop<decltype(f1),Real,fftw_format>(shrinkingMask.image,autocorrelation,&f1,1);
      shrinkingMask.cpyToGM();
    }
    if(setups.doCentral || padAutoCorrelation){
      targetfft = fftw(autocorrelation, targetfft, 1, setups.forwardFactor);
    }
    if(1){ //apply random phase
      for(int i = 0; i<row*column; i++){ 
        complex<Real> &data = ((fftw_format*)targetfft->data)[i];
        Real mod = (setups.doCentral||padAutoCorrelation)? sqrt(abs(data)) : abs(data);
        if(setups.useBS && ((Real*)beamStop.image->data)[i]>0.5) {
          data = 0.;
        }
        else{
          Real randphase = static_cast<Real>(rand())/RAND_MAX*2*pi;
          data = mod*exp(complex<Real>(0,randphase));
        }
      }
    }

    convertFromComplexToInteger<Real>(beamStop.image, cache,MOD,0);
    imwrite("mask.png",*cache);
    //auto f = [&](int x, int y, fftw_format &data){
    //  auto tmp = (complex<Real>*)&data;
    //  *tmp = 1.4+*tmp;
    //};
    //imageLoop<decltype(f)>(autocorrelation,&f,0);
    if(!setups.restart){
      fftresult = new Mat();
      targetfft->copyTo(*fftresult);
    }
    convertFromComplexToInteger(targetfft, cache, PHASE,1);
    imwrite("init_phase.png",*cache);
    convertFromComplexToInteger(targetfft, cache, MOD2,1,1,"Pattern MOD2");
    imwrite("init_pattern.png",*cache);
    convertFromComplexToInteger(targetfft, cache, MOD2,1,1,"Pattern MOD2",1);
    imwrite("init_logpattern.png",*cache);
    //if(setups.runSim) targetfft = convertFromIntegerToComplex(*cache, targetfft, 1 , "waveFront");
    convertFromComplexToInteger(autocorrelation, cache, REAL,1,1,"Autocorrelation MOD2",1);
    imwrite("auto_correlation.png",*cache);
    //Phase retrieving starts from here. In the following, only targetfft is needed.
    if(setups.doIteration) phaseRetrieve(setups, targetfft, gkp1, cache, fftresult); //fftresult is the starting point of the iteration
    //Now let's do KCDI


    return 0;
}

