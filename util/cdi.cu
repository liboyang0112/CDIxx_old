#include <complex>
#include <tbb/tbb.h>
#include <fftw3-mpi.h>
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

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

//#define Bits 16
const dim3 threadsPerBlock(16,16);
__device__ __constant__ double cuda_beta_HIO;
__device__ __constant__ int cuda_row;
__device__ __constant__ int cuda_column;
__device__ __constant__ int cuda_rcolor;
__device__ __constant__ double cuda_scale;
__device__ __constant__ int cuda_totalIntensity;
cufftHandle *plan = new cufftHandle();
using namespace cv;
double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

double gaussian_norm(double x, double y, double sigma){
  return 1./(2*pi*sigma*sigma)*gaussian(x,y,sigma);
}

/******************************************************************************/

void maskOperation(Mat &input, Mat &output, Mat &kernel){
  filter2D(input, output, input.depth(), kernel);
}

class support{
public:
  support(){};
  __device__ __host__ virtual bool isInside(int x, int y) = 0;
};
class ImageMask{
public:
  int nrow;
  int ncol;
  size_t sz;
  double *data;
  Mat *image;
  double threshold;
  ImageMask *cuda;
  ImageMask(){
    cudaMalloc(&cuda, sizeof(ImageMask));
  };
  void init_image(Mat* image_){
    nrow = image_->rows;
    ncol = image_->cols;
    image = image_;
    sz = image_->total()*sizeof(double);
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
	    //printf("%d, %d = %f lower than threshold, dropping\n",x,y,image->ptr<double>(x)[y]);
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
  double r;
  __device__ __host__ bool isInside(int x, int y){
    double dr = sqrt(pow(x-x0,2)+pow(y-y0,2));
    if(dr < r) return true;
    return false;
  }
};

template <typename sptType>
__global__ void createMask(double* data, sptType* spt, bool isFrequency=0){
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

Mat* gaussianKernel(int rows, int cols, double sigma){
  Mat* image = new Mat(rows, cols, CV_64FC1);
  auto f = [&](int x, int y, double &data){
    data = gaussian_norm(x-rows/2,y-cols/2,sigma);
  };
  imageLoop<decltype(f), double>(image,&f);
  
  return image;
}
/******************************************************************************/
__global__ void applyNorm(cufftDoubleComplex* data){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  data[index].x*=1./sqrtf(cuda_row*cuda_column);
  data[index].y*=1./sqrtf(cuda_row*cuda_column);
}

__global__ void applyMod(cufftDoubleComplex* source, cufftDoubleComplex* target, ImageMask *bs = 0, bool loose=0){
  assert(source!=0);
  assert(target!=0);
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  if(loose && bs && bs->isInside(x,y)) return;
  double tolerance = 0.5/cuda_rcolor*cuda_scale*16+30./cuda_rcolor; // fluctuation caused by bit depth and noise
  double maximum = pow(mergeDepth,2)*cuda_scale*0.99;
  int index = x*cuda_column + y;
  cufftDoubleComplex targetdata = target[index];
  cufftDoubleComplex sourcedata = source[index];
  double ratiox = 1;
  double ratioy = 1;
  double mod2 = targetdata.x*targetdata.x + targetdata.y*targetdata.y;
  double srcmod2 = sourcedata.x*sourcedata.x + sourcedata.y*sourcedata.y;
  if(mod2>=maximum) {
    if(loose) mod2 = max(maximum,srcmod2);
    else tolerance*=1000;
  }
  double diff = mod2-srcmod2;
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

Mat* createWaveFront(Mat &intensity, Mat &phase, int rows, int columns, Mat* &itptr, Mat* wavefront = 0){
  if ( !intensity.data )
  {
      printf("No intensity data \n");
      exit(0);
  }
  if ( !phase.data )
  {
      printf("No phase data \n");
      exit(0);
  }
  if(intensity.rows!=phase.rows || intensity.cols!=phase.cols) {
    printf("intensity map and phase map having different dimensions");
    exit(0);
  }
  columns = intensity.cols;
  rows = intensity.rows;
  Mat *imageptr;
  itptr = &intensity;
  Mat &intensity_sc = *itptr;
  if(phase.channels()==3){
    imageptr = new Mat(rows, columns, format_cv);
    cv::cvtColor(phase, *imageptr, cv::COLOR_BGR2GRAY);
  }else{
    imageptr = &phase;
  }
  Mat &phase_sc = *imageptr;
  //wavefront = convertFromIntegerToComplex(intensity_sc, wavefront,0,"waveFront");
  wavefront = convertFromIntegerToComplex(intensity_sc, phase_sc, wavefront);
  delete imageptr;
  return wavefront;
  //imwrite("input.png",image);
}


void ApplyERSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhoprime){
  if(insideS){
    rhonp1[0] = rhoprime[0];
    rhonp1[1] = rhoprime[1];
  }else{
    rhonp1[0] = rhonp1[1] = 0;
  }
}
void ApplyPOSERSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhoprime){
  if(insideS){
    rhonp1[0] = rhoprime[0]*( rhoprime[0] > 0 );
    rhonp1[1] = 0;
  }else{
    rhonp1[0] = rhonp1[1] = 0;
  }
}
void ApplyLoosePOSERSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhoprime, double threshold){
  if(rhoprime[0] < threshold){
    rhonp1[0] = rhoprime[0]*( rhoprime[0] > 0 );
  }else{
    rhonp1[0] = threshold;
  }
    rhonp1[1] = 0;
}
__device__ void ApplyHIOSupport(bool insideS, cufftDoubleComplex &rhonp1, cufftDoubleComplex &rhoprime, double beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
    rhonp1.y -= beta*rhoprime.y;
  }
}
__device__ void ApplyPOSHIOSupport(bool insideS, cufftDoubleComplex &rhonp1, cufftDoubleComplex &rhoprime, double beta){
  if(rhoprime.x > 0 && (insideS/* || rhoprime[0]<30./rcolor*/)){
    rhonp1.x = rhoprime.x;
  }else{
    rhonp1.x -= beta*rhoprime.x;
  }
  rhonp1.y -= beta*rhoprime.y;
}
void ApplyLoosePOSHIOSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhoprime, double beta, double threshold){
  if(rhoprime[0] > 0 && (rhoprime[0]<threshold)){
    rhonp1[0] = rhoprime[0];
    //rhonp1[1] = rhoprime[1];
    rhonp1[1] -= beta*rhoprime[1];
  }else{
    rhonp1[0] -= beta*(rhoprime[0]);
    rhonp1[1] -= beta*rhoprime[1];
  }
}
void ApplySFSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhoprime){
  if(insideS){
    rhonp1[0] = rhoprime[0];
    rhonp1[1] = rhoprime[1];
  }else{
    rhonp1[0] = -0.9*rhoprime[0];
    rhonp1[1] = -0.9*rhoprime[1];
  }
}
void ApplyDMSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhop, fftw_complex &pmsrho, double gammas, double gammam, double beta){

  complex<double> &rho = *(complex<double>*)rhonp1;
  complex<double> &rhoprime = *(complex<double>*)rhop;
  complex<double> &pmpsrho = *(complex<double>*)pmsrho;
  if(1||insideS){
    rho = 2.*pmpsrho-rhoprime;//(1-beta*gammam)*rhoprime+beta*(1+gammam+gammas)*rhoprime-beta*(1+gammas)*pmpsrho;
  }else{
    rho += 2.*pmpsrho-rhoprime;//beta*gammas*rhoprime-beta*(1+gammas)*pmpsrho;
  }
}
void ApplyPOSDMSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhop, fftw_complex &pmsrho, double gammas, double gammam, double beta){

  complex<double> rho(rhonp1[0],rhonp1[1]);
  complex<double> rhoprime(rhop[0],rhop[1]);
  complex<double> pmpsrho(pmsrho[0],pmsrho[1]);
  if(insideS){
    rho = (1-beta*gammam)*rhoprime+beta*(1+gammam+gammas)*rhoprime-beta*(1+gammas)*pmpsrho;
  }else{
    rho += beta*gammas*rhoprime-beta*(1+gammas)*pmpsrho;
  }
  rhonp1[0] = rho.real();
  if(rhonp1[0]<0) rhonp1[0] = 0;
  rhonp1[1] = 0;
}
struct experimentConfig{
 bool useDM;
 bool useBS;
 bool useShrinkMap = 0;
 bool reconAC = 0;
 ImageMask* spt;
 ImageMask* beamStop;
 bool restart;
 double lambda = 0.6;
 double d = 16e3;
 double pixelsize = 6.5;
 double beamspotsize = 50;
};

__global__ void applySupport(cufftDoubleComplex *gkp1, cufftDoubleComplex *gkprime, double* objMod, ImageMask *spt){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = x*cuda_column + y;
  if(x >= cuda_row || y >= cuda_column) return;

  //epsilonF+=hypot(gkp1data[0]-gkprimedata[0],gkp1data[1]-gkprimedata[1]);
  //fftw_complex tmp = {gkp1data[0],gkp1data[1]};
  bool inside = spt->isInside(x,y);
  cufftDoubleComplex &gkp1data = gkp1[index];
  cufftDoubleComplex &gkprimedata = gkprime[index];
  //if(iter >= niters - 20 ) ApplyERSupport(inside,gkp1data,gkprimedata);
  //if(iter >= niters - 20 || iter % 200 == 0) ApplyERSupport(inside,gkp1data,gkprimedata);
  //if(iter >= niters - 20 || iter<20) ApplyERSupport(inside,gkp1data,gkprimedata);
  //if(iter >= niters - 20) ApplyERSupport(inside,gkp1data,gkprimedata);
  //ApplyERSupport(inside,gkp1data,gkprimedata);
  //else ApplyHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
  //else ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
  //printf("%d, (%f,%f), (%f,%f), %f\n",inside, gkprimedata.x,gkprimedata.y,gkp1data.x,gkp1data.y,cuda_beta_HIO);
  ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  objMod[index] = cuCabs(gkp1data);
  //double thres = gaussian(x-row/2,y-column/2,40);
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
void phaseRetrieve( experimentConfig &setups, Mat* targetfft, Mat* gkp1 = 0, Mat *cache = 0, Mat* fftresult = 0 ){
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
    if(gkp1==0) gkp1 = new Mat(row,column,CV_64FC2);
    assert(targetfft!=0);
    double beta = -1;
    double beta_HIO = 0.9;
    cudaMemcpyToSymbol(cuda_beta_HIO,&beta_HIO,sizeof(beta_HIO));
    double gammas = -1./beta;
    double gammam = 1./beta;
    double epsilonS, epsilonF;
    std::ofstream fepF,fepS;
    fepF.open("epsilonF.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    fepS.open("epsilonS.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    int niters = 5000;
    int tot = row*column;
    bool saveIter=1;
    Mat objMod(row,column,CV_64FC1);
    Mat* maskKernel;
    double gaussianSigma = 3;

    size_t sz = row*column*sizeof(cufftDoubleComplex);
    cufftDoubleComplex *cuda_fftresult, *cuda_targetfft, *cuda_gkprime, *cuda_gkp1, *cuda_pmpsg;
    double *cuda_objMod;
    ImageMask *cuda_spt;
    cudaMalloc((void**)&cuda_fftresult, sz);
    cudaMalloc((void**)&cuda_targetfft, sz);
    cudaMalloc((void**)&cuda_gkprime, sz);
    cudaMalloc((void**)&cuda_gkp1, sz);
    cudaMalloc((void**)&cuda_objMod, sz/2);
    cudaMalloc((void**)&cuda_spt, sizeof(ImageMask));
    cudaMemcpy(cuda_spt, &re, sizeof(ImageMask), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_targetfft, targetfft->data, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_fftresult, fftresult->data, sz, cudaMemcpyHostToDevice);

    dim3 numBlocks((row-1)/threadsPerBlock.x+1, (column-1)/threadsPerBlock.y+1);
    //dim3 numBlocks(row/threadsPerBlock.x, column/threadsPerBlock.y);

    cufftExecZ2Z( *plan, cuda_targetfft, cuda_gkp1, CUFFT_INVERSE);
    applyNorm<<<numBlocks,threadsPerBlock>>>(cuda_gkp1);
    cudaDeviceSynchronize();
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
	  time_FFT.count(),     ((double)time_FFT.count())/tot*100,
	  time_norm.count(),    ((double)time_norm.count())/tot*100,
	  time_support.count(), ((double)time_support.count())/tot*100,
	  time_applyMod.count(),((double)time_applyMod.count())/tot*100
        );
	if(saveIter){
          cudaMemcpy(gkp1->data, cuda_gkp1, sz, cudaMemcpyDeviceToHost);
          convertFromComplexToInteger( gkp1,cache, MOD2,0);
          std::string iterstr = to_string(iter);
          imwrite("recon_intensity"+iterstr+".png",*cache);
          convertFromComplexToInteger( gkp1,cache, PHASE,0);
          imwrite("recon_phase"+iterstr+".png",*cache);
	}
      }
      now = std::chrono::high_resolution_clock::now();
      if(useBS) applyMod<<<numBlocks,threadsPerBlock>>>(cuda_fftresult,cuda_targetfft,cir.cuda, !setups.reconAC || iter > 1000);  //apply mod to fftresult, Pm
      else applyMod<<<numBlocks,threadsPerBlock>>>(cuda_fftresult,cuda_targetfft,0, !setups.reconAC || iter > 1000);  //apply mod to fftresult, Pm
      if(useDM) {
        if(useBS) applyMod<<<numBlocks,threadsPerBlock>>>(cuda_pmpsg,cuda_targetfft,cir.cuda, !setups.reconAC || iter > 1000);  
        else applyMod<<<numBlocks,threadsPerBlock>>>(cuda_pmpsg,cuda_targetfft,0, !setups.reconAC || iter > 1000);
      }
      //cudaDeviceSynchronize();
      time_applyMod+=std::chrono::high_resolution_clock::now()-now;
      
      epsilonS = epsilonF = 0;
      now = std::chrono::high_resolution_clock::now();
      cufftExecZ2Z( *plan, cuda_fftresult, cuda_gkprime, CUFFT_INVERSE);
      time_FFT+=std::chrono::high_resolution_clock::now()-now;
      now = std::chrono::high_resolution_clock::now();
      applyNorm<<<numBlocks,threadsPerBlock>>>(cuda_gkprime);
     // cudaDeviceSynchronize();
      time_norm+=std::chrono::high_resolution_clock::now()-now;
      if(useDM){
        now = std::chrono::high_resolution_clock::now();
        cufftExecZ2Z( *plan, cuda_pmpsg, cuda_pmpsg, CUFFT_INVERSE);
        time_FFT+=std::chrono::high_resolution_clock::now()-now;
        now = std::chrono::high_resolution_clock::now();
        applyNorm<<<numBlocks,threadsPerBlock>>>(cuda_pmpsg);
        cudaDeviceSynchronize();
        time_norm+=std::chrono::high_resolution_clock::now()-now;
      }
      bool updateMask = (iter > 1000) && iter%20==0 && useShrinkMap && iter!=0;
      if(updateMask){
        int size = floor(gaussianSigma*6); // r=3 sigma to ensure the contribution is negligible (0.01 of the maximum)
        size = size/2*2+1; //ensure odd
        maskKernel = gaussianKernel(size,size,gaussianSigma);
      }
      now = std::chrono::high_resolution_clock::now();
      applySupport<<<numBlocks,threadsPerBlock>>>(cuda_gkp1, cuda_gkprime, cuda_objMod,cuda_spt);
      time_support+=std::chrono::high_resolution_clock::now()-now;
      /*
        cudaMemcpy(gkp1->data, cuda_gkp1, sz, cudaMemcpyDeviceToHost);
        convertFromComplexToInteger( gkp1,cache, MOD2,0);
        imwrite("debug.png",*cache);
	*/
      //cudaDeviceSynchronize();
      if(updateMask){
        cudaMemcpy(objMod.data, cuda_objMod, sz/2, cudaMemcpyDeviceToHost);
        filter2D(objMod, *re.image,objMod.depth(),*maskKernel);
	((ImageMask*)&re)->cpyToGM();
	if(gaussianSigma>1.5) gaussianSigma*=0.99;
	delete maskKernel;
      }
      if(updateMask&&iter%100==0&&saveIter){
	convertFromComplexToInteger<double>(re.image, cache,MOD,0);
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
      cufftExecZ2Z( *plan, cuda_gkp1, cuda_fftresult, CUFFT_FORWARD);
      time_FFT+=std::chrono::high_resolution_clock::now()-now;
      now = std::chrono::high_resolution_clock::now();
      applyNorm<<<numBlocks,threadsPerBlock>>>(cuda_fftresult);
      //cudaDeviceSynchronize();
      time_norm+=std::chrono::high_resolution_clock::now()-now;
      if(useDM){ // FFT to get f field;
        cufftExecZ2Z( *plan, cuda_pmpsg, cuda_pmpsg, CUFFT_FORWARD);
        applyNorm<<<numBlocks,threadsPerBlock>>>(cuda_pmpsg);
        cudaDeviceSynchronize();
      }
      //end iteration
    }
    fepF.close();
    fepS.close();
    cudaMemcpy(fftresult->data, cuda_fftresult, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(targetfft->data, cuda_targetfft, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(gkp1->data, gkp1, sz, cudaMemcpyDeviceToHost);

    convertFromComplexToInteger( gkp1,cache, MOD2,0);
    imwrite("recon_intensity.png",*cache);
    convertFromComplexToInteger(gkp1, cache, PHASE,0);
    imwrite("recon_phase.png",*cache);
    if(useDM)  convertFromComplexToInteger( pmpsg, cache, MOD2,1);
    if(useDM)  imwrite("recon_pmpsg.png",*cache);
    convertFromComplexToInteger( fftresult, cache, MOD2,1);
    imwrite("recon_pattern.png",*cache);
}

__global__ void applyAutoCorrelationMod(cufftDoubleComplex* source,cufftDoubleComplex* target, ImageMask *bs = 0){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column + y;
  double targetdata = target[index].x;
  double retval = targetdata;
  source[index].y = 0;
  double maximum = pow(mergeDepth,2)*cuda_scale*0.99;
  double sourcedata = source[index].x;
  double tolerance = 0.5/cuda_rcolor*cuda_scale;
  double diff = sourcedata-targetdata;
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
__global__ void applyERACSupport(cufftDoubleComplex* data,cufftDoubleComplex* prime,sptType *spt, double* objMod){
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
  objMod[index] = cuCabs(data[index]);
}

template <typename sptType>
__global__ void applyHIOACSupport(cufftDoubleComplex* data,cufftDoubleComplex* prime, sptType *spt, double *objMod){
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
  objMod[index] = cuCabs(data[index]);
}

template <typename sptType>
void autoCorrelationConstrain(Mat* pattern, sptType *spt, Mat* cache, ImageMask *bs = 0){  //beam stop
  double totalIntensity = 1000;
  cudaMemcpyToSymbol(cuda_totalIntensity,&totalIntensity,sizeof(totalIntensity));
  dim3 numBlocks((pattern->rows-1)/threadsPerBlock.x+1, (pattern->cols-1)/threadsPerBlock.y+1);
  cufftDoubleComplex *autocorrelation, *cuda_pattern, *autoprime;
  ImageMask autoCorrelationMask;
  ImageMask *cuda_Mask;
  autoCorrelationMask.init_image(new Mat(pattern->rows,pattern->cols,CV_64FC1));
  autoCorrelationMask.threshold = 0.008;
  sptType *cuda_spt;
  size_t sz = pattern->total()*sizeof(cufftDoubleComplex);
  double *cuda_objMod;
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
  convertFromComplexToInteger<double>(autoCorrelationMask.image, cache,MOD, 1);
  imwrite("autoCorrelationMask.png",*cache);
  cufftExecZ2Z(*plan,cuda_pattern,autocorrelation,CUFFT_INVERSE);
  applyNorm<<<numBlocks,threadsPerBlock>>>(autocorrelation);
  cudaMemcpy(pattern->data, autocorrelation, sz, cudaMemcpyDeviceToHost);
  convertFromComplexToInteger(pattern, cache, REAL, 1);
  imwrite("initAC.png",*cache);
  double gaussianSigma=3;
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
    cufftExecZ2Z(*plan,autocorrelation,autoprime,CUFFT_FORWARD);
    applyNorm<<<numBlocks,threadsPerBlock>>>(autoprime);
    applyAutoCorrelationMod<<<numBlocks,threadsPerBlock>>>(autoprime, cuda_pattern, bs?bs->cuda:0);
    cufftExecZ2Z(*plan,autoprime,autoprime,CUFFT_INVERSE);
    applyNorm<<<numBlocks,threadsPerBlock>>>(autoprime);
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

    if(argc < 2){
      printf("please feed the object intensity and phase image\n");
    }
    bool runSim;
    bool simCCDbit = 1;
    printf("command:");
    for(int i = 0; i < argc ; i++){
	    printf("%s ",argv[i]);
    }
    printf("\n");
    if(argv[1] == std::string("sim")){
      runSim = 1;
    }else{
      runSim = 0;
    }


    auto seed = (unsigned)time(NULL);
    bool isFresnel = 0;
    bool doIteration = 1;
    bool useGaussionLumination = 0;
    bool useGaussionHERALDO = 0;
    bool useRectHERALDO = 0;

    //int seed = 1657180926;// 1657180330
    //int seed = 1657182238; // oversampling = 3, modulation range = pi, reversed image
    //1657182741 // oversampling = 3, modulation range = 1.1pi, reversed image
    //1657182948 // oversampling = 3, modulation range = 1.3pi, reversed image
    //1657184141 // oversampling = 3, modulation range = 2pi, upright image, random phase
    srand(seed);
    printf("seed:%d\n",seed);
    double oversampling = 4;
    Mat* gkp1 = 0;
    Mat* targetfft = 0;
    Mat* fftresult = 0;
    bool restart = 0;
    if(argc > 4){
      restart = 1;
      
    }
    int row, column;
    Mat intensity = readImage( argv[2] , !runSim);
    //maskOperation(intensity,intensity);
    Mat ele = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));
    //erode( intensity, intensity, ele);
    //dilate( intensity, intensity, ele);
    row = intensity.rows;
    column = intensity.cols;
    pixeltype *rowp;
    if(useRectHERALDO){
      for(int i = 0; i < row ; i++){
        rowp = intensity.ptr<pixeltype>(i);
        for(int j = 0; j < column ; j++){
          if(i > row/2 || j > column/2) rowp[j] = rcolor-1;
	}
      }
    }


    if(runSim){
	    row*=oversampling;
	    column*=oversampling;
    }
    cudaMemcpyToSymbol(cuda_row,&row,sizeof(row));
    cudaMemcpyToSymbol(cuda_column,&column,sizeof(column));
    cudaMemcpyToSymbol(cuda_rcolor,&rcolor,sizeof(rcolor));
    //double tmp = scale*10;
    cudaMemcpyToSymbol(cuda_scale,&scale,sizeof(scale));

    cufftPlan2d ( plan, row, column, CUFFT_Z2Z);

    C_circle cir,cir2,cir3;
    //cir is the beam stop
    //cir.x0=row/2-50;
    //cir.y0=column/2+20;
    //cir.r=50;
    cir.x0=row/2;
    cir.y0=column/2;
    cir.r=10;
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
    re.startx = (oversampling-1)/2*row/oversampling;
    re.starty = (oversampling-1)/2*column/oversampling;
    //re.startx = 1./4*row;
    //re.starty = 1./4*column;
    re.endx = row-re.startx;
    re.endy = column-re.starty;
    
    dim3 numBlocks((row-1)/threadsPerBlock.x+1, (column-1)/threadsPerBlock.y+1);

    experimentConfig setups;
    setups.reconAC = 0;
    setups.useShrinkMap = 1;
    ImageMask shrinkingMask;
    shrinkingMask.threshold = 0.1;
    setups.useDM = 0;

    setups.useBS = 1;
    ImageMask beamStop;
    beamStop.threshold = 0.5;
    beamStop.init_image(new Mat(row,column,CV_64FC1));
    C_circle *cuda_spt;
    cudaMalloc((void**)&cuda_spt,sizeof(C_circle));
    cudaMemcpy(cuda_spt, &cir, sizeof(C_circle), cudaMemcpyHostToDevice);
    createMask<<<numBlocks,threadsPerBlock>>>(beamStop.data, cuda_spt,1);
    beamStop.cpyFromGM();
    cudaFree(cuda_spt);
    beamStop.updateCuda();

    setups.spt = &shrinkingMask;
    //setups.spt = &re;
    //setups.spt = &cir3;
    
    setups.beamStop = &beamStop;//&cir;
    setups.restart = restart;
    //setups.d = oversampling*setups.pixelsize*setups.beamspotsize/setups.lambda; //distance to guarentee oversampling
    setups.pixelsize = 7;//setups.d/oversampling/setups.beamspotsize*setups.lambda;
    printf("recommanded imaging distance = %f\n", setups.d);
    printf("recommanded pixel size = %f\n", setups.pixelsize);

    bool isFarField = 0;
    double reversefresnelNumber = setups.d*setups.lambda/pi/pow(setups.beamspotsize,2);
    printf("Fresnel Number = %f\n",1./reversefresnelNumber);
    if(reversefresnelNumber > 100) isFarField = 1;
    //these are for simulation
    Mat* cache = 0;
    Mat* cache1;
    cache1 = &intensity;
    if(runSim){
      if(argc==4){
        Mat phase = readImage( argv[3]);
        if(oversampling>1) 
          gkp1 = createWaveFront(*extend(*cache1,oversampling), *extend(phase,oversampling), row, column,cache,gkp1);
        //if(oversampling>1) gkp1 = createWaveFront(extend(intensity,oversampling), extend(phase,oversampling), row, column,cache,gkp1);
        else gkp1 = createWaveFront(intensity,phase, row, column,cache,gkp1);
      }else{
        if(oversampling>1) cache = extend(*cache1,oversampling);
	else cache = cache1;
	gkp1 = convertFromIntegerToComplex(*cache, gkp1,0,"waveFront");
      }
      if(!isFarField && isFresnel){
        auto f = [&](int x, int y, fftw_complex &data){
          auto tmp = (complex<double>*)&data;
	  double phase = pi*setups.lambda*setups.d/pow(setups.pixelsize,2)*(pow((x-0.5*row)/row,2)+pow((y-0.5*column)/column,2))/10;
	  *tmp *= exp(complex<double>(0,phase));
	};
        imageLoop<decltype(f)>(gkp1,&f,0);
      }
      if(useGaussionLumination){
        //setups.spt = &re;
        //if(!setups.useShrinkMap) setups.spt = &cir3;
        //diffraction image, either from simulation or from experiments.
        auto f = [&](int x, int y, fftw_complex &data){
          auto tmp = (complex<double>*)&data;
          bool inside = cir3.isInside(x,y);
	  if(!inside) *tmp = 0.;
	  *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir3.r);
	  //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
	};
        imageLoop<decltype(f)>(gkp1,&f,0);
      }
      if(useGaussionHERALDO){
        auto f = [&](int x, int y, fftw_complex &data){
          auto tmp = (complex<double>*)&data;
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
      targetfft = fftw(gkp1,targetfft,1); 
    }else{
      //if(mergeDepth == 1) cache = cache1;
      //else 
      cache = new Mat(row, column, format_cv);
      if(cache1->depth() == CV_64F) 
        targetfft = cache1;
      else
        targetfft = convertFromIntegerToComplex(*cache1,targetfft,1); 
    }
    if(restart){
      intensity = readImage(argv[3]);
      Mat phase = readImage(argv[4]);
      gkp1 = createWaveFront(intensity, phase, row, column,cache,gkp1);
      fftresult = fftw(gkp1,fftresult,1); //If not restart, this line just allocate space, the values are not used.
    }
    //cir2.x0=row/2;
    //cir2.y0=column/2;
    double decay = scale;
    if(runSim) decay=1;
    std::default_random_engine generator;
    double noiseLevel = 9;
    std::poisson_distribution<int> distribution(noiseLevel);
    Mat *autocorrelation = new Mat(row,column,CV_64FC2,Scalar::all(0.));
    shrinkingMask.init_image(new Mat(row,column,CV_64FC1));
    for(int i = 0; i<row*column; i++){ //remove the phase information
     // double randphase = arg(tmp);//static_cast<double>(rand())/RAND_MAX*2*pi;
      complex<double> &data = *(complex<double>*)((fftw_complex*)targetfft->data)[i];
      fftw_complex &datacor = ((fftw_complex*)autocorrelation->data)[i];
      double mod = abs(data)*sqrt(decay);
      if(runSim&&simCCDbit) {
        int range= pow(2,12);
        mod = sqrt(((double)floor(pow(mod,2)*range))/(range)); //assuming we use 16bit CCD
        mod = sqrt(max(0.,pow(mod,2)+double(distribution(generator)-noiseLevel)/range)); //Poisson noise
      }
      if(1){
        if(setups.useBS && ((double*)beamStop.image->data)[i]>0.5) {
          data = 0.;
	  mod = 0;
        }
        else{
          //complex<double> tmp(targetfft[i][0],targetfft[i][1]);
          double randphase = static_cast<double>(rand())/RAND_MAX*2*pi;
          data = mod*exp(complex<double>(0,randphase));
        }
      }
      //datacor[0] = pow(mod,2)*(tx-row/2)*(ty-column/2)/90; // ucore is the derivitaves of the diffraction pattern: append *(tx-row/2)*(ty-column/2)/20;
      datacor[0] = pow(mod,2); //ucore is the diffraction pattern
      datacor[1] = 0;
    }
    convertFromComplexToInteger( autocorrelation, cache, REAL,1,1,"HERALDO U core"); 
    imwrite("ucore.png",*cache);
    if(!setups.reconAC || runSim) {
      autocorrelation = fftw(autocorrelation, autocorrelation, 0);
      rect *cuda_spt;
      cudaMalloc((void**)&cuda_spt,sizeof(rect));
      cudaMemcpy(cuda_spt, &re, sizeof(rect), cudaMemcpyHostToDevice);
      createMask<<<numBlocks,threadsPerBlock>>>(shrinkingMask.data, cuda_spt,0);
      shrinkingMask.cpyFromGM();
      cudaFree(cuda_spt);
      shrinkingMask.updateCuda();
    }
    else {
      autoCorrelationConstrain(autocorrelation, &re, cache, setups.useBS?&beamStop:0);
      fftw(autocorrelation, targetfft, 1);
      auto f =  [&](int x, int y, fftw_complex &data){
	data[0] = sqrt(max(data[0],0.));
	data[1] = 0;
      };
      imageLoop<decltype(f)>(targetfft, &f, 0);
      auto f1 = [&](int x, int y, double &data, fftw_complex &dataout){
        data = hypot(dataout[1],dataout[0])>shrinkingMask.threshold;
      };
      imageLoop<decltype(f1),double,fftw_complex>(shrinkingMask.image,autocorrelation,&f1,1);
      shrinkingMask.cpyToGM();
    }

    convertFromComplexToInteger<double>(shrinkingMask.image, cache,MOD,0);
    imwrite("mask.png",*cache);
    //auto f = [&](int x, int y, fftw_complex &data){
    //  auto tmp = (complex<double>*)&data;
    //  *tmp = 1.4+*tmp;
    //};
    //imageLoop<decltype(f)>(autocorrelation,&f,0);
    if(!restart){
      fftresult = new Mat();
      targetfft->copyTo(*fftresult);
    }
    convertFromComplexToInteger(targetfft, cache, PHASE,1);
    imwrite("init_phase.png",*cache);
    convertFromComplexToInteger(targetfft, cache, MOD2,1,1,"Pattern MOD2");
    imwrite("init_pattern.png",*cache);
    //if(runSim) targetfft = convertFromIntegerToComplex(*cache, targetfft, 1 , "waveFront");
    convertFromComplexToInteger(autocorrelation, cache, REAL,1,1,"Autocorrelation MOD2",1);
    imwrite("auto_correlation.png",*cache);
    //Phase retrieving starts from here. In the following, only targetfft is needed.
    if(doIteration) phaseRetrieve(setups, targetfft, gkp1, cache, fftresult); //fftresult is the starting point of the iteration
    return 0;
}
