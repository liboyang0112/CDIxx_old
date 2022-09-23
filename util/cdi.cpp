#include <complex>
#include <tbb/parallel_for.h>
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

#include "common.h"
#include "imageReader.h"

//static tbb::affinity_partitioner ap;

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
//#define Bits 16
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
  virtual bool isInside(int x, int y) = 0;
};
class ImageMask : public support{
public:
  Mat *image; //rows, cols, CV_64FC1
  ImageMask():support(){};
  double threshold;
  bool isInside(int x, int y){
    if(image->ptr<double>(x)[y] < threshold) {
	    //printf("%d, %d = %f lower than threshold, dropping\n",x,y,image->ptr<double>(x)[y]);
	    return false;
    }
    return true;
  }
};
class rect : public support{
public:
  int startx;
  int starty;
  int endx;
  int endy;
  rect():support(){};
  bool isInside(int x, int y){
    if(x > startx && x <= endx && y > starty && y <= endy) return true;
    return false;
  }
};
class C_circle : public support{
public:
  int x0;
  int y0;
  double r;
  C_circle():support(){};
  bool isInside(int x, int y){
    double dr = sqrt(pow(x-x0,2)+pow(y-y0,2));
    if(dr < r) return true;
    return false;
  }
};
Mat* gaussianKernel(int rows, int cols, double sigma){
  Mat* image = new Mat(rows, cols, CV_64FC1);
  auto f = [&](int x, int y, double &data){
    data = gaussian_norm(x-rows/2,y-cols/2,sigma);
  };
  imageLoop<decltype(f), double>(image,&f);
  return image;
}

void applyMod(Mat* source, Mat* target, support *bs = 0){
  assert(source!=0);
  assert(target!=0);
  double tolerance = 0.5/rcolor*scale;
  double maximum = pow(mergeDepth,2)*scale;
  int row = target->rows;
  int column = target->cols;
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, row),
    [&](const tbb::blocked_range<size_t> r)
    {
      for (size_t x = r.begin(); x != r.end(); ++x)
      {
        fftw_complex* rowo,*rowp;
        rowo = source->ptr<fftw_complex>(x);
        rowp = target->ptr<fftw_complex>(x);
        for(size_t y = 0; y < column; y++){
	  if(bs!=0){
            int tx = x;
            if(x >= row/2) tx -= row/2;
            else tx += row/2;
            int ty = y;
            if(y >= column/2) ty -= column/2;
            else ty += column/2;
            if(bs->isInside(tx,ty)) {
              continue;
            }
          }
          fftw_complex &targetdata = rowp[y];
          fftw_complex &sourcedata = rowo[y];
          double ratio = 1;
          double mod2 = targetdata[0]*targetdata[0] + targetdata[1]*targetdata[1];
          double srcmod2 = sourcedata[0]*sourcedata[0] + sourcedata[1]*sourcedata[1];
          if(mod2>=maximum) {
            mod2 = max(maximum,srcmod2);
          }
          if(srcmod2 == 0){
            sourcedata[0] = sqrt(mod2);
            sourcedata[1] = 0;
            continue;
          }
          double diff = mod2-srcmod2;
          if(diff>tolerance){
            ratio = sqrt((mod2-tolerance)/srcmod2);
          }else if(diff < -tolerance ){
            ratio = sqrt((mod2+tolerance)/srcmod2);
          }
          sourcedata[0] *= ratio;
          sourcedata[1] *= ratio;
        }
      }
    }
  );
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
void ApplyHIOSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhoprime, double beta){
  if(insideS){
    rhonp1[0] = rhoprime[0];
    rhonp1[1] = rhoprime[1];
  }else{
    rhonp1[0] -= beta*rhoprime[0];
    rhonp1[1] -= beta*rhoprime[1];
  }
}
void ApplyPOSHIOSupport(bool insideS, fftw_complex &rhonp1, fftw_complex &rhoprime, double beta){
  if(rhoprime[0] > 0 && (insideS/* || rhoprime[0]<30./rcolor*/)){
    rhonp1[0] = rhoprime[0];
    //rhonp1[1] = rhoprime[1];
    rhonp1[1] -= beta*rhoprime[1];
  }else{
    rhonp1[0] -= beta*rhoprime[0];
    rhonp1[1] -= beta*rhoprime[1];
  }
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
 bool useShrinkMap = 1;
 support* spt;
 support* beamStop;
 bool restart;
 double lambda = 0.6;
 double d = 16e3;
 double pixelsize = 6.5;
 double beamspotsize = 50;
};

void phaseRetrieve( experimentConfig &setups, Mat* targetfft, Mat* gkp1 = 0, Mat *cache = 0, Mat* fftresult = 0 ){
    Mat* pmpsg = 0;
    bool useShrinkMap = setups.useShrinkMap;
    int row = targetfft->rows;
    int column = targetfft->cols;
    bool useDM = setups.useDM;
    bool useBS = setups.useBS;
    auto &re = *(setups.spt);
    auto &cir = *(setups.beamStop);
    if(useDM) {
      pmpsg = new Mat();
      fftresult->copyTo(*pmpsg);
    }
    Mat* gkprime = 0;
    assert(targetfft!=0);
    double beta = -1;
    double beta_HIO = 0.9;
    double gammas = -1./beta;
    double gammam = 1./beta;
    double epsilonS, epsilonF;
    gkp1 = fftw(targetfft,gkp1,0); //IFFT to get O field;
    std::ofstream fepF,fepS;
    fepF.open("epsilonF.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    fepS.open("epsilonS.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    int niters = 5000;
    int tot = row*column;
    bool saveIter=1;
    Mat objMod(row,column,CV_64FC1);
    Mat* maskKernel;
    double gaussianSigma = 3;
    for(int iter = 0; iter < niters; iter++){
      //start iteration
      if(iter%100==0 && saveIter) {
        printf("Iteration Number : %d\n", iter);
        convertFromComplexToInteger( gkp1,cache, MOD2,0);
        std::string iterstr = to_string(iter);
        imwrite("recon_intensity"+iterstr+".png",*cache);
        convertFromComplexToInteger( gkp1,cache, PHASE,0);
        imwrite("recon_phase"+iterstr+".png",*cache);
      }
      if(useBS) applyMod(fftresult,targetfft,&cir);  //apply mod to fftresult, Pm
      else applyMod(fftresult,targetfft);  //apply mod to fftresult, Pm
      if(useDM) {
        if(useBS) applyMod(pmpsg,targetfft,&cir);  
	else applyMod(pmpsg,targetfft);
      }
      epsilonS = epsilonF = 0;
      gkprime = fftw(fftresult,gkprime,0);
      if(useDM) pmpsg = fftw(pmpsg,pmpsg,0);
      bool updateMask = iter%20==0 && useShrinkMap && iter!=0;
      if(updateMask){
        int size = floor(gaussianSigma*6); // r=3 sigma to ensure the contribution is negligible (0.01 of the maximum)
        size = size/2*2+1; //ensure odd
        maskKernel = gaussianKernel(size,size,gaussianSigma);
      }
      tbb::parallel_for(
        tbb::blocked_range<size_t>(0, row),
        [&](const tbb::blocked_range<size_t> &r)
        {
          for (int x = r.begin(); x != r.end(); ++x){
	    fftw_complex *gkp1p = gkp1->ptr<fftw_complex>(x);
	    fftw_complex *gkprimep = gkprime->ptr<fftw_complex>(x);
	    double *objModp;
	    if(updateMask){
              objModp = objMod.ptr<double>(x);
	    }
            for (int y = 0; y < column; ++y){
	      fftw_complex &gkp1data = gkp1p[y];
	      fftw_complex &gkprimedata = gkprimep[y];
              epsilonF+=hypot(gkp1data[0]-gkprimedata[0],gkp1data[1]-gkprimedata[1]);
              fftw_complex tmp = {gkp1data[0],gkp1data[1]};
              bool inside = re.isInside(x,y);
              //if(iter >= niters - 20 ) ApplyERSupport(inside,gkp1data,gkprimedata);
              //if(iter >= niters - 20 || iter % 200 == 0) ApplyERSupport(inside,gkp1data,gkprimedata);
              //if(iter >= niters - 20 || iter<20) ApplyERSupport(inside,gkp1data,gkprimedata);
              //if(iter >= niters - 20) ApplyERSupport(inside,gkp1data,gkprimedata);
              //ApplyERSupport(inside,gkp1data,gkprimedata);
              //else ApplyHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
              //else ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
	      ApplyHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
	      if(updateMask){
                objModp[y] = hypot(gkp1data[0],gkp1data[1]);
	      }
	      //double thres = gaussian(x-row/2,y-column/2,40);
	      //ApplyLoosePOSHIOSupport(inside,gkp1data,gkprimedata,beta_HIO,thres);
              //ApplyLoosePOSERSupport(inside,gkp1data,gkprimedata,thres);
              //else {
              //ApplyDMSupport(inside,gkp1data, gkprimedata, pmpsg[index], gammas, gammam, beta);
              //}
              //ApplyERSupport(inside,pmpsg[index],gkp1data);
              //ApplyHIOSupport(inside,gkp1data,gkprimedata,beta);
              //else ApplySFSupport(inside,gkp1data,gkprimedata);
              epsilonS+=hypot(tmp[0]-gkp1data[0],tmp[1]-gkp1data[1]);
	    }
	  }
        }
      );
      if(updateMask){
        filter2D(objMod, *((ImageMask*)&re)->image,objMod.depth(),*maskKernel);
	if(gaussianSigma>1.5) gaussianSigma*=0.99;
	delete maskKernel;
      }
      if(updateMask&&iter%100==0&&saveIter){
	convertFromComplexToInteger<double>(((ImageMask*)&re)->image, cache,MOD,0);
	//convertFromComplexToInteger<double>(&objMod, cache,MOD,0);
        std::string iterstr = to_string(iter);
	imwrite("mask"+iterstr+".png",*cache);
      }
      if(iter!=0){
        fepF<<sqrt(epsilonF/tot)<<endl;
        fepS<<sqrt(epsilonS/tot)<<endl;
      }else {
        convertFromComplexToInteger( gkp1,cache, MOD2,0);
        imwrite("recon_support.png",*cache);
        convertFromComplexToInteger( gkp1,cache, PHASE,0);
        imwrite("recon_phase_support.png",*cache);
      }

      //if(sqrt(epsilonS/row/column)<0.05) break;
      fftresult = fftw(gkp1,fftresult,1); // FFT to get f field;
      if(useDM) pmpsg = fftw(pmpsg,pmpsg,1); // FFT to get f field;
      //end iteration
    }
    fepF.close();
    fepS.close();

    convertFromComplexToInteger( gkp1,cache, MOD2,0);
    imwrite("recon_intensity.png",*cache);
    convertFromComplexToInteger(gkp1, cache, PHASE,0);
    imwrite("recon_phase.png",*cache);
    if(useDM)  convertFromComplexToInteger( pmpsg, cache, MOD2,1);
    if(useDM)  imwrite("recon_pmpsg.png",*cache);
    convertFromComplexToInteger( fftresult, cache, MOD2,1);
    imwrite("recon_pattern.png",*cache);
}
/*
void autoCorrelationConstrain(Mat* pattern, support spt){
  Mat* autocorrelation = 0;
  for(int i = 0; i < 1000; i++){
    autocorrelation = fftw(pattern, autocorrelation, 0);
    for(int x = 0; x <pattern->row ; x++){
      patter->ptr
      spt.isInside(x,y);
    }
}
*/

int main(int argc, char** argv )
{

    if(argc < 2){
      printf("please feed the object intensity and phase image\n");
    }
    bool runSim;
    bool simCCDbit = 0;
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
    bool useGaussionLumination = 1;
    bool useGaussionHERALDO = 0;
    bool useRectHERALDO = 0;

    //int seed = 1657180926;// 1657180330
    //int seed = 1657182238; // oversampling = 3, modulation range = pi, reversed image
    //1657182741 // oversampling = 3, modulation range = 1.1pi, reversed image
    //1657182948 // oversampling = 3, modulation range = 1.3pi, reversed image
    //1657184141 // oversampling = 3, modulation range = 2pi, upright image, random phase
    srand(seed);
    printf("seed:%d\n",seed);
    double oversampling = 3;
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
    C_circle cir,cir2,cir3,cir4;
    cir.x0=row/2;
    cir.y0=column/2;
    cir.r=10;
    //cir2.x0 = column*2/3-50;
    //cir2.y0 = row*2/3+110;
    //cir2.r = 50;
    cir2.x0 = column/2;
    cir2.y0 = row/2;
    cir2.r = 20;
    cir3.x0 = row/2;
    cir3.y0 = column/2;
    //cir3.r = 300/mergeDepth;
    cir3.r = 20;
    cir4.x0 = cir2.x0;
    cir4.y0 = cir2.y0;
    cir4.r = cir3.r;
    rect re;
    re.startx = (oversampling-1)/2*row/oversampling;
    re.starty = (oversampling-1)/2*column/oversampling;
    //re.startx = 1./4*row;
    //re.starty = 1./4*column;
    re.endx = row-re.startx;
    re.endy = column-re.starty;
    

    experimentConfig setups;
    setups.useShrinkMap = 1;
    ImageMask shrinkingMask;
    shrinkingMask.threshold = 0.1;
    setups.useDM = 0;
    setups.useBS = 0;

    setups.spt = &shrinkingMask;
    //setups.spt = &re;
    //setups.spt = &cir3;
    
    setups.beamStop = &cir;
    setups.restart = restart;
    //setups.d = oversampling*setups.pixelsize*setups.beamspotsize/setups.lambda; //distance to guarentee oversampling
    setups.pixelsize = 7;//setups.d/oversampling/setups.beamspotsize*setups.lambda;
    printf("recommanded imaging distance = %f\n", setups.d);
    printf("recommanded pixel size = %f\n", setups.pixelsize);

    bool isFarField = 0;
    double reversefresnelNumber = setups.d*setups.lambda/pi/pow(setups.beamspotsize,2);
    printf("Fresnel Number = %f\n",1./reversefresnelNumber);
    if(reversefresnelNumber > 100) isFarField = 1;
    size_t sz = row*column*sizeof(fftw_complex);
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
        if(!setups.useShrinkMap) setups.spt = &cir3;
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
    std::poisson_distribution<int> distribution(1000);
    Mat *autocorrelation = new Mat(row,column,CV_64FC2,Scalar::all(0.));
    shrinkingMask.image = new Mat(row,column,CV_64FC1);
    for(int i = 0; i<row*column; i++){ //remove the phase information
     // double randphase = arg(tmp);//static_cast<double>(rand())/RAND_MAX*2*pi;
      int tx = i/row;
      if(tx >= row/2) tx -= row/2;
      if(i/row < row/2) tx += row/2;
      int ty = i%row;
      if(ty >= column/2) ty -= column/2;
      if(i%row < column/2) ty += column/2;
      complex<double> &data = *(complex<double>*)((fftw_complex*)targetfft->data)[i];
      fftw_complex &datacor = ((fftw_complex*)autocorrelation->data)[i];
      double mod = abs(data)*sqrt(decay);
      if(runSim&&simCCDbit) {
        int range= pow(2,16);
        mod = sqrt(((double)floor(pow(mod,2)*range))/(range)); //assuming we use 16bit CCD
        //mod = sqrt(pow(mod,2)+double(distribution(generator))/range); //Poisson noise
      }
      if(1){
      if(setups.useBS && cir.isInside(tx,ty)) {
        data = 0.;
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
    convertFromComplexToInteger( autocorrelation, cache, MOD,1,1,"HERALDO U core"); 
    imwrite("ucore.png",*cache);
    autocorrelation = fftw(autocorrelation, autocorrelation, 0);
    //autoCorrelationReconstruction(autocorrelation);

    auto f = [&](int x, int y, double &data, fftw_complex &dataout){
      data = hypot(dataout[1],dataout[0])>shrinkingMask.threshold;
    };
    imageLoop<decltype(f),double,fftw_complex>(shrinkingMask.image,autocorrelation,&f,1);
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
    if(runSim) targetfft = convertFromIntegerToComplex(*cache, targetfft, 1 , "waveFront");
    convertFromComplexToInteger(autocorrelation, cache, MOD2,1,1,"Autocorrelation MOD2",1);
    imwrite("auto_correlation.png",*cache);
    //Phase retrieving starts from here. In the following, only targetfft is needed.
    if(doIteration) phaseRetrieve(setups, targetfft, gkp1, cache, fftresult); //fftresult is the starting point of the iteration
    return 0;
}
