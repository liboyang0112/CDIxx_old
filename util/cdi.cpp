#include <complex>
#include <tbb/tbb.h>
#include <fftw3-mpi.h>
//# include <stdlib.h>
# include <cassert>
# include <stdio.h>
# include <time.h>
# include <random>

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <libconfig.h++>


// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
//#define Bits 16
#define Bits 16


static tbb::affinity_partitioner ap;
using namespace cv;
// Declare the variables
using namespace std;
static const int mergeDepth = 1;
#if Bits==12
using pixeltype=uint16_t;
static const int nbits = 12;
static const auto format_cv = CV_16UC1;
#elif Bits==16
using pixeltype=uint16_t;
static const int nbits = 16;
static const auto format_cv = CV_16UC1;
#else
using pixeltype=uchar;
static const int nbits = 8;
static const auto format_cv = CV_8UC1;
#endif
//using inputtype=uchar;
//static const int inputbits = 8;

static const int rcolor = pow(2,nbits);
static bool opencv_reverted = 0;

const double pi = 3.1415927;
using namespace cv;
double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
  //return 1./sqrt(2*pi*sigma)*exp(-r2/2/pow(sigma,2));
}

template<typename T>
T sqrsum(int cnt, T val){
  cnt++;
  return val*val;
}
template<typename T, typename... Ts>
T sqrsum(int &cnt, T val, Ts... vals){
  cnt++;
  return val*val+sqrsum(cnt,vals...);
}
template<typename... Ts>
double rms(Ts... vals){
  int n = 1;
  double sum = sqrsum(n, vals...);
  return sqrt(sum/n);
}
enum mode {MOD2,MOD, REAL, IMAG, PHASE};
/******************************************************************************/

void maskOperation(Mat &input, Mat &output){
  Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                     -1,  5, -1,
                                      0, -1,  0);
  filter2D(input, output, input.depth(), kernel);
}

class support{
public:
  support(){};
  virtual bool isInside(int x, int y) = 0;
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
template<typename functor>
void imageLoop(Mat* data, void* arg, bool isFrequency = 0){
  int row = data->rows;
  int column = data->cols;
  fftw_complex *rowp;
  functor* func = static_cast<functor*>(arg);
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = data->ptr<fftw_complex>(x);
    for(int y = 0; y<column; y++){
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      (*func)(targetx, targety , rowp[y]);
    }
  }
}
Mat* fftw ( Mat* in, Mat *out = 0, bool isforward = 1)
{
  fftw_plan plan_forward;
  int row = in->rows;
  int column = in->cols;
  double ratio = 1./sqrt(row*column);
  if(out == 0) out = new Mat(row,column,CV_64FC2);

  plan_forward = fftw_plan_dft_2d ( row, column, (fftw_complex*)in->data, (fftw_complex*)out->data, isforward?FFTW_FORWARD:FFTW_BACKWARD, FFTW_ESTIMATE );

  fftw_execute ( plan_forward );

  for(int i = 0; i < row*column ; i++){
    ((fftw_complex*)out->data)[i][0]*=ratio;
    ((fftw_complex*)out->data)[i][1]*=ratio;
  }
/*only used to check the correctness of the transformation
  fftw_plan plan_backward
  plan_backward = fftw_plan_dft_2d ( row, column, out, in, FFTW_BACKWARD, FFTW_ESTIMATE );

  fftw_execute ( plan_backward );

  for(int i = 0; i < row*column ; i++){
    in[i][0]*=ratio;
    in[i][1]*=ratio;
  }
  fftw_destroy_plan ( plan_backward );
*/
  fftw_destroy_plan ( plan_forward );

  return out;
}
/******************************************************************************/
template <typename inputtype>
Mat read16bitImage(Mat imagein, int nbitsimg)
{
    int row = imagein.rows;
    int column = imagein.cols;
    int threshold = 1;
    int factor = pow(2,16-nbitsimg);
    Mat image(row, column, CV_16UC(1), Scalar::all(0));
    inputtype* rowp;
    uint16_t* rowo;
    int tot = 0;
    int max = 0;
    for(int x = 0; x < row ; x++){
	rowp = imagein.ptr<inputtype>(x);
	rowo = image.ptr<uint16_t>(x);
	//rowo = image.ptr<char>(x);
        for(int y = 0; y<column; y++){
            //rowp[y] = (int)(255*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.25));
	    //if(rowp[y]>0) rowo[y]=rowp[y]/256;//log2(rowp[y])*pow(2,11);
	    //int nm1 = rowp[y-1];
	    //int np1 = rowp[y+1];
	    //if(n!=0) {
    	    //  int score0 = 4; // noise filter
	    //  if(y==0 || nm1<=threshold) score0--;
	    //  if(y==column-1 || np1<=threshold) score0--;
	    //  if(x==0 || imagein.ptr<inputtype>(x-1)[y]<=threshold) score0--;
	    //  if(x==row-1 || imagein.ptr<inputtype>(x+1)[y]<=threshold) score0--;
	    //  if(score0 > 1 || rowo[y] > threshold){
            //    //rowo[y]=floor(log2(n)*pow(2,12));//log2(rowp[y])*pow(2,11);
            //    if(nbits > inputbits) rowo[y]=n<<(nbits-inputbits);//log2(rowp[y])*pow(2,11);
	    //    else rowo[y]=n>>(inputbits-nbits);//log2(rowp[y])*pow(2,11);
	    //  }
	    //}
	    rowo[y] = rowp[y]*factor;
	    tot+= rowp[y];
	    if(max < rowp[y]) max = rowp[y];
	}
    }
    printf("\ntot=%d,max=%d\n",tot,max);
    return image;
}

Mat readImage(char* name){
  Mat imagein = imread( name, IMREAD_UNCHANGED  );
  if(nbits == 8) {
   if(imagein.channels()==3){
     Mat image(imagein.rows, imagein.cols, format_cv);
     cv::cvtColor(imagein, image, cv::COLOR_BGR2GRAY);
     return image;
   }else{
     return imagein;
   }
  }
  if(imagein.depth() == CV_8U){
    return read16bitImage<char>(imagein,8);
  }else{
    return read16bitImage<uint16_t>(imagein,16);
  }
}

Mat* convertFromOpencvToFFTW(Mat &image, Mat* cache = 0, bool isFrequency = 0, const char* label= "default"){
  int row = image.rows/mergeDepth*mergeDepth;
  int column = image.cols/mergeDepth*mergeDepth;
  if(!cache) cache = new Mat(row/mergeDepth, column/mergeDepth, CV_64FC2);
  else if(mergeDepth!=1) *cache = Scalar(0.,0.);
  double tot = 0;
  pixeltype* rowp;
  fftw_complex* rowo;
  int targetx, targety;
  for(int x = 0; x < row ; x++){
    if(isFrequency){
      targetx = x<row/2?x+row/2:(x-row/2);
    }else{
      targetx = x;
    }
    rowp = image.ptr<pixeltype>(x);
    rowo = cache->ptr<fftw_complex>(targetx/mergeDepth);
    for(int y = 0; y<column; y++){
      if(isFrequency){
        targety = y<column/2?y+column/2:(y-column/2);
      }else{
	targety = y;
      }
      double intensity = ((double)rowp[y])/(rcolor-1);
      fftw_complex &datatmp = rowo[targety/mergeDepth];
      if(opencv_reverted) intensity = 1-intensity;
      if(mergeDepth==1) datatmp[0] = sqrt(intensity);
      else{
        datatmp[0] = sqrt(pow(datatmp[0],2)+intensity);
      }
      //cache[index][0] = sqrt(intensity);
      datatmp[1] = 0;
      tot += sqrt(intensity);
    }
  }
  printf("total intensity %s: %f\n",label, tot);
  return cache;
}

Mat* convertFromOpencvToFFTW(Mat &image,Mat &phase,Mat* cache = 0){
  int row = image.rows;
  int column = image.cols;
  if(!cache) cache = new Mat(row, column, CV_64FC2);
  int tot = 0;
  pixeltype *rowi, *rowp;
  fftw_complex *rowo;
  for(int x = 0; x < row ; x++){
    rowi = image.ptr<pixeltype>(x);
    rowp = phase.ptr<pixeltype>(x);
    rowo = phase.ptr<fftw_complex>(x);
    for(int y = 0; y<column; y++){
      tot += rowp[y];
      double phase = rowp[y];
      //phase*=2*pi/rcolor;
      //phase-=pi;
      phase = static_cast<double>(rand())/RAND_MAX*2*pi;
      rowo[y][0] = sqrt(((double)rowi[y])/rcolor)*cos(phase);
      rowo[y][1] = sqrt(((double)rowi[y])/rcolor)*sin(phase);
    }
  }
  printf("total intensity: %d\n", tot);
  return cache;
}

void convertFromFFTWToOpencv(Mat &image, Mat* cache, mode m, bool isFrequency, double decay = 1, const char* label= "default"){
  pixeltype* rowp;
  fftw_complex* rowo;
  int row = cache->rows;
  int column = cache->cols;
  int tot = 0;
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = image.ptr<pixeltype>(targetx);
    rowo = cache->ptr<fftw_complex>(x);
    for(int y = 0; y<column; y++){
      double target;
      complex<double> &tmpc = *(complex<double>*)(rowo[y]);
      switch(m){
        case MOD:
          target = std::abs(tmpc);
          break;
        case MOD2:
          target = pow(std::abs(tmpc),2);
          break;
        case IMAG:
          target = tmpc.imag();
          break;
        case PHASE:
	  if(std::abs(tmpc)==0) target = 0;
	  else target = (std::arg(tmpc)+pi)/2/pi;
          break;
        default:
          target = tmpc.real();
      }
      target*=rcolor*decay;
      tot += (int)target;
      if(target<0) target = -target;
      if(target>=rcolor) {
	      //printf("larger than maximum of %s png %f\n",label, target);
	      target=rcolor-1;
	      //target=0;
      }
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      rowp[targety] = floor(target);
      //if(opencv_reverted) rowp[targety] = rcolor - 1 - rowp[targety];
      //rowp[targety] = rcolor - 1 - rowp[targety];
    }
  }
  printf("total intensity %s: %d\n", label, tot);
}
void applyMod(Mat* source, Mat* target, support *bs = 0){
  assert(source!=0);
  assert(target!=0);
  double tolerance = 0.5/rcolor;
  double maximum = pow(mergeDepth,2);
  int row = target->rows;
  int column = target->cols;
  parallel_for(
    tbb::detail::d1::blocked_range<size_t>(0, row),
    [&](const tbb::detail::d1::blocked_range<size_t> &r)
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
              //printf("skipping %d, %d, %d\n",tx, ty, i);
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
    },
    ap
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
  //wavefront = convertFromOpencvToFFTW(intensity_sc, wavefront,0,"waveFront");
  wavefront = convertFromOpencvToFFTW(intensity_sc, phase_sc, wavefront);
  delete imageptr;
  return wavefront;
  //imwrite("input.png",image);
}


Mat* extend( Mat &src , double ratio, double val = 0)
{
  Mat *dst = new Mat();
  int top, bottom, left, right;
  int borderType = BORDER_CONSTANT;
  if( src.empty()) {
      printf(" Error opening image\n");
      printf(" Program Arguments: [image_name -- default lena.jpg] \n");
      exit(0);
  }
  // Initialize arguments for the filter
  top = (int) ((ratio-1)/2*src.rows); bottom = top;
  left = (int) ((ratio-1)/2*src.cols); right = left;
  Scalar value(opencv_reverted?rcolor:0);
  copyMakeBorder( src, *dst, top, bottom, left, right, borderType, value );
  imwrite("ext.png",*dst);
  return dst;
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
  if(insideS && rhoprime[0] > 0){
    rhonp1[0] = rhoprime[0];
    //rhonp1[1] = rhoprime[1];
    rhonp1[1] -= beta*rhoprime[1];
  }else{
    rhonp1[0] -= beta*rhoprime[0];
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
    for(int iter = 0; iter < niters; iter++){                                                            
      //start iteration
      if(iter%100==0) {
        printf("Iteration Number : %d\n", iter);
        convertFromFFTWToOpencv(*cache, gkp1, MOD2,0);
        std::string iterstr = to_string(iter);
        imwrite("recon_intensity"+iterstr+".png",*cache);
        convertFromFFTWToOpencv(*cache, gkp1, PHASE,0);
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
      parallel_for(
        tbb::detail::d1::blocked_range<size_t>(0, row),
        [&](const tbb::detail::d1::blocked_range<size_t> &r)
        {
          for (size_t x = r.begin(); x != r.end(); ++x){
	    fftw_complex *gkp1p = gkp1->ptr<fftw_complex>(x);
	    fftw_complex *gkprimep = gkprime->ptr<fftw_complex>(x);
            for (size_t y = 0; y < column; ++y){
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
	      ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,beta_HIO);
              //else {
              //ApplyDMSupport(inside,gkp1data, gkprimedata, pmpsg[index], gammas, gammam, beta);
              //}
              //ApplyERSupport(inside,pmpsg[index],gkp1data);
              //ApplyHIOSupport(inside,gkp1data,gkprimedata,beta);
              //else ApplySFSupport(inside,gkp1data,gkprimedata);
              epsilonS+=rms(tmp[0]-gkp1data[0],tmp[1]-gkp1data[1]);
	    }
	  }
        },ap);
      if(iter>=1){
        fepF<<sqrt(epsilonF/tot)<<endl;
        fepS<<sqrt(epsilonS/tot)<<endl;
      }
      if(iter==0) {
        convertFromFFTWToOpencv(*cache, gkp1, MOD2,0);
        imwrite("recon_support.png",*cache);
        convertFromFFTWToOpencv(*cache, gkp1, PHASE,0);
        imwrite("recon_phase_support.png",*cache);
      }

      //if(sqrt(epsilonS/row/column)<0.05) break;
      fftresult = fftw(gkp1,fftresult,1); // FFT to get f field;
      if(useDM) pmpsg = fftw(pmpsg,pmpsg,1); // FFT to get f field;
      //end iteration
    }
    fepF.close();
    fepS.close();

    convertFromFFTWToOpencv(*cache, gkp1, MOD2,0);
    imwrite("recon_intensity.png",*cache);
    convertFromFFTWToOpencv(*cache, gkp1, PHASE,0);
    imwrite("recon_phase.png",*cache);
    if(useDM)  convertFromFFTWToOpencv(*cache, pmpsg, MOD2,1);
    if(useDM)  imwrite("recon_pmpsg.png",*cache);
    convertFromFFTWToOpencv(*cache, fftresult, MOD2,1);
    imwrite("recon_pattern.png",*cache);
}

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
    bool useGaussionLumination = 0;
    bool useGaussionHERALDO = 0;
    bool useRectHERALDO = 0;
    fftw_init_threads();
    fftw_plan_with_nthreads(3);

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
    Mat intensity = readImage( argv[2] );
    //maskOperation(intensity,intensity);
    Mat ele = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));
    //erode( intensity, intensity, ele);
    //dilate( intensity, intensity, ele);
    row = intensity.rows/mergeDepth;
    column = intensity.cols/mergeDepth;
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
    cir2.r = 40;
    cir3.x0 = row/2;
    cir3.y0 = column/2;
    //cir3.r = 300/mergeDepth;
    cir3.r = 40;
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
    setups.useDM = 0;
    setups.useBS = 0;
    setups.spt = &re;
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
	gkp1 = convertFromOpencvToFFTW(*cache, gkp1,0,"waveFront");
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
        setups.spt = &cir3;
        //diffraction image, either from simulation or from experiments.
        auto f = [&](int x, int y, fftw_complex &data){
          auto tmp = (complex<double>*)&data;
          bool inside = cir2.isInside(x,y);
	  if(!inside) *tmp = 0.;
	  *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2);
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
      convertFromFFTWToOpencv(*cache, gkp1, MOD2,0,1,"Object MOD2");
      imwrite("init_object.png",*cache);
      convertFromFFTWToOpencv(*cache, gkp1, PHASE,0,1,"Object Phase");
      imwrite("init_object_phase.png",*cache);
      targetfft = fftw(gkp1,targetfft,1); 
    }else{
      if(mergeDepth == 1) cache = cache1;
      else cache = new Mat(row, column, format_cv);
      targetfft = convertFromOpencvToFFTW(*cache1,targetfft,1); 
    }
    if(restart){
      intensity = readImage(argv[3]);
      Mat phase = readImage(argv[4]);
      gkp1 = createWaveFront(intensity, phase, row, column,cache,gkp1);
      fftresult = fftw(gkp1,fftresult,1); //If not restart, this line just allocate space, the values are not used.
    }
    //cir2.x0=row/2;
    //cir2.y0=column/2;
    double decay = 0.9;
    std::default_random_engine generator;
    std::poisson_distribution<int> distribution(1000);
    Mat *autocorrelation = new Mat(row,column,CV_64FC2);
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
      datacor[0] = pow(mod,2)*(tx-row/2)*(ty-column/2)/20;
      datacor[1] = 0;
    }
    convertFromFFTWToOpencv(*cache, autocorrelation, MOD,1,1,"HERALDO U core");
    imwrite("ucore.png",*cache);
    autocorrelation = fftw(autocorrelation, autocorrelation, 0);
        auto f = [&](int x, int y, fftw_complex &data){
          auto tmp = (complex<double>*)&data;
	  *tmp = 1.4+*tmp;
	};
        imageLoop<decltype(f)>(autocorrelation,&f,0);
    if(!restart){
      fftresult = new Mat();
      targetfft->copyTo(*fftresult);
    }
    convertFromFFTWToOpencv(*cache, targetfft, PHASE,1);
    imwrite("init_phase.png",*cache);
    convertFromFFTWToOpencv(*cache, targetfft, MOD2,1,1,"Pattern MOD2");
    imwrite("init_pattern.png",*cache);
    targetfft = convertFromOpencvToFFTW(*cache, targetfft , 1 , "waveFront");
    convertFromFFTWToOpencv(*cache, autocorrelation, MOD2,1,1,"Autocorrelation MOD2");
    imwrite("auto_correlation.png",*cache);
    //Phase retrieving starts from here. In the following, only targetfft is needed.
    if(doIteration) phaseRetrieve(setups, targetfft, gkp1, cache, fftresult); //fftresult is the starting point of the iteration
    return 0;
}
