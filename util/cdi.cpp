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
  int n = 0;
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
void imageLoop(fftw_complex* data, int row, int column, void* arg, bool isFrequency = 0){
  functor* func = static_cast<functor*>(arg);
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    for(int y = 0; y<column; y++){
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      (*func)(targetx, targety , data[targetx*column+targety]);
    }
  }
}
fftw_complex* fftw ( fftw_complex* in, int row, int column, fftw_complex *out = 0, bool isforward = 1)
{
  fftw_plan plan_forward;
  double ratio = 1./sqrt(row*column);
  if(out == 0) out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );

  plan_forward = fftw_plan_dft_2d ( row, column,  in, out, isforward?FFTW_FORWARD:FFTW_BACKWARD, FFTW_ESTIMATE );

  fftw_execute ( plan_forward );

  for(int i = 0; i < row*column ; i++){
    out[i][0]*=ratio;
    out[i][1]*=ratio;
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

fftw_complex* convertFromOpencvToFFTW(Mat image, fftw_complex* cache = 0, bool isFrequency = 0, const char* label= "default"){
  int row = image.rows/mergeDepth*mergeDepth;
  int column = image.cols/mergeDepth*mergeDepth;
  size_t sz = sizeof ( fftw_complex ) * (row/mergeDepth) * (column/mergeDepth);
  if(!cache) cache = (fftw_complex*) fftw_malloc ( sz );
  memset(cache, 0, sz);
  double tot = 0;
  pixeltype* rowp;
  int targetx, targety;
  for(int x = 0; x < row ; x++){
    rowp = image.ptr<pixeltype>(x);
    for(int y = 0; y<column; y++){
      if(isFrequency){
        targetx = x<row/2?x+row/2:(x-row/2);
        targety = y<column/2?y+column/2:(y-column/2);
      }else{
        targetx = x;
	targety = y;
      }
      double intensity = ((double)rowp[y])/(rcolor-1);
      if(opencv_reverted) intensity = 1-intensity;
      int index = targetx/mergeDepth*(column/mergeDepth)+targety/mergeDepth;
      cache[index][0] = sqrt(pow(cache[index][0],2)+intensity);
      //cache[index][0] = sqrt(intensity);
      cache[index][1] = 0;
      tot += sqrt(intensity);
    }
  }
  printf("total intensity %s: %f\n",label, tot);
  return cache;
}

fftw_complex* convertFromOpencvToFFTW(Mat &image,Mat phase,fftw_complex* cache = 0){
  int row = image.rows;
  int column = image.cols;
  if(!cache) cache = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
  int tot = 0;
  pixeltype *rowi, *rowp;
  for(int x = 0; x < row ; x++){
    rowi = image.ptr<pixeltype>(x);
    rowp = phase.ptr<pixeltype>(x);
    for(int y = 0; y<column; y++){
      tot += rowp[y];
      double phase = rowp[y];
      //phase*=2*pi/rcolor;
      //phase-=pi;
      phase = static_cast<double>(rand())/RAND_MAX*2*pi;
      cache[x*column+y][0] = sqrt(((double)rowi[y])/rcolor)*cos(phase);
      cache[x*column+y][1] = sqrt(((double)rowi[y])/rcolor)*sin(phase);
    }
  }
  printf("total intensity: %d\n", tot);
  return cache;
}

void convertFromFFTWToOpencv(Mat &image, int row, int column, fftw_complex* cache, mode m, bool isFrequency, double decay = 1, const char* label= "default"){
  pixeltype* rowp;
  int tot = 0;
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = image.ptr<pixeltype>(targetx);
    for(int y = 0; y<column; y++){
      double target;
      complex<double> &tmpc = *(complex<double>*)(cache[x*column+y]);
      switch(m){
        case MOD:
          target = std::abs(tmpc);
          break;
        case MOD2:
          target = pow(std::abs(tmpc),2);
          break;
        case REAL:
          target = cache[x*column+y][0];
          break;
        case IMAG:
          target = cache[x*column+y][1];
          break;
        case PHASE:
	  if(std::abs(tmpc)==0) target = 0;
	  else target = (std::arg(tmpc)+pi)/2/pi;
          break;
        default:
          target = cache[x*column+y][0];
      }
      target*=rcolor*decay;
      tot += (int)target;
      if(target<0) target = -target;
      if(target>=rcolor) {
	      //printf("larger than maximum of %s png %f\n",label, target);
	      target=rcolor-1;
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
void applyMod(fftw_complex* source, fftw_complex* target, int row, int column){
  assert(source!=0);
  assert(target!=0);
  double tolerance = 0.5/rcolor;
  double maximum = pow(mergeDepth,2);
  parallel_for(
    tbb::detail::d1::blocked_range<size_t>(0, row*column),
    [&](const tbb::detail::d1::blocked_range<size_t> &r)
    {
      for (size_t i = r.begin(); i != r.end(); ++i)
      {
        double ratio = 1;
        double mod2 = target[i][0]*target[i][0] + target[i][1]*target[i][1];
        double srcmod2 = source[i][0]*source[i][0] + source[i][1]*source[i][1];
        if(mod2>=maximum) {
          mod2 = max(maximum,srcmod2);
        }
        if(srcmod2 == 0){
          source[i][0] = sqrt(mod2);
          source[i][1] = 0;
          continue;
        }
        double diff = mod2-srcmod2;
        if(diff>tolerance){
          ratio = sqrt((mod2-tolerance)/srcmod2);
        }else if(diff < -tolerance ){
          ratio = sqrt((mod2+tolerance)/srcmod2);
        }
        source[i][0] *= ratio;
        source[i][1] *= ratio;
      }
    },
    ap
  );
}
void applyModBS(fftw_complex* source, fftw_complex* target, int row, int column, support &bs){
  assert(source!=0);
  assert(target!=0);
  for(int i = 0; i < row*column ; i++){
    int tx = i/row;
    if(tx >= row/2) tx -= row/2;
    if(i/row < row/2) tx += row/2;
    int ty = i%row;
    if(ty >= column/2) ty -= column/2;
    if(i%row < column/2) ty += column/2;
    if(bs.isInside(tx,ty)) {
      //printf("skipping %d, %d, %d\n",tx, ty, i);
      continue;
    }
    double mod = rms(target[i][0],target[i][1]);
    double phase;
    complex<double> tmpc(source[i][0],source[i][1]);
    phase = std::arg(tmpc);
    source[i][0] = cos(phase)*mod;
    source[i][1] = sin(phase)*mod;
  }
}
void createGaussion(int row, int column, fftw_complex* &inputField, fftw_complex* &targetField){
  inputField = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
  targetField = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
  for(int x = 0; x < row ; x++){
    for(int y = 0; y<column; y++){
      double Emod =  3*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.1);
      double Emodt = 3*gaussian(((double)x)/row-0.1,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.2,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.3,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.4,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.6,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.7,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.8,((double)y)/column-0.5,0.01)
                   + 3*gaussian(((double)x)/row-0.9,((double)y)/column-0.5,0.01);
      double randphase = static_cast<double>(rand())/RAND_MAX*2*pi;
      inputField[x*column+y][0] = Emod*sin(randphase);
      inputField[x*column+y][1] = Emod*cos(randphase);
      int targetx = x<row/2?x+row/2:(x-row/2);
      int targety = y<column/2?y+column/2:(y-column/2);
      targetField[targetx*column+targety][0] = Emodt*sin(randphase);
      targetField[targetx*column+targety][1] = Emodt*cos(randphase);
    }
  }
}
fftw_complex* createWaveFront(Mat &intensity, Mat &phase, int rows, int columns, Mat* &itptr, fftw_complex* wavefront = 0){
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

void phaseRetrieve( experimentConfig &setups, int row, int column, fftw_complex* targetfft, fftw_complex* gkp1 = 0, Mat *cache = 0, fftw_complex* fftresult = 0 ){
    fftw_complex* pmpsg = 0;
    bool useDM = setups.useDM;
    bool useBS = setups.useBS;
    auto &re = *(setups.spt);
    auto &cir = *(setups.beamStop);
    if(useDM) {
      size_t sz = row*column*sizeof(fftw_complex);
      pmpsg = (fftw_complex*) malloc(sz);//just to allocate space, the values are not used.
      memcpy(pmpsg, fftresult,sz);
    }
    fftw_complex* gkprime = 0;
    assert(targetfft!=0);
    double beta = -1;
    double beta_HIO = 0.9;
    double gammas = -1./beta;
    double gammam = 1./beta;
    double epsilonS, epsilonF;
    gkp1 = fftw(targetfft,row,column,gkp1,0); //IFFT to get O field;
    std::ofstream fepF,fepS;
    fepF.open("epsilonF.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    fepS.open("epsilonS.txt",ios::out |(setups.restart? ios::app:std::ios_base::openmode(0)));
    int niters = 5000;
    for(int iter = 0; iter < niters; iter++){                                                            
      //start iteration
      if(iter%100==0) {
        printf("Iteration Number : %d\n", iter);
        convertFromFFTWToOpencv(*cache,row, column, gkp1, MOD2,0);
        std::string iterstr = to_string(iter);
        imwrite("recon_intensity"+iterstr+".png",*cache);
        convertFromFFTWToOpencv(*cache,row, column, gkp1, PHASE,0);
        imwrite("recon_phase"+iterstr+".png",*cache);
      }
      if(useBS) applyModBS(fftresult,targetfft,row,column,cir);  //apply mod to fftresult, Pm
      else applyMod(fftresult,targetfft,row,column);  //apply mod to fftresult, Pm
      if(useDM) {
        if(useBS) applyModBS(pmpsg,targetfft,row,column,cir);  
	else applyMod(pmpsg,targetfft,row,column);
      }
      epsilonS = epsilonF = 0;
      gkprime = fftw(fftresult,row,column,gkprime,0);
      if(useDM) pmpsg = fftw(pmpsg,row,column,pmpsg,0);
      int index = 0;
      parallel_for(
        tbb::detail::d1::blocked_range<size_t>(0, row*column),
        [&](const tbb::detail::d1::blocked_range<size_t> &r)
        {
          for (size_t index = r.begin(); index != r.end(); ++index){
            epsilonF+=rms(gkp1[index][0]-gkprime[index][0],gkp1[index][1]-gkprime[index][1]);
            fftw_complex tmp = {gkp1[index][0],gkp1[index][1]};
            bool inside = re.isInside(index/row,index%row);
            //if(iter >= niters - 20 ) ApplyERSupport(inside,gkp1[index],gkprime[index]);
            //if(iter >= niters - 20 || iter % 200 == 0) ApplyERSupport(inside,gkp1[index],gkprime[index]);
            //if(iter >= niters - 20 || iter<20) ApplyERSupport(inside,gkp1[index],gkprime[index]);
            //if(iter >= niters - 20) ApplyERSupport(inside,gkp1[index],gkprime[index]);
            //ApplyERSupport(inside,gkp1[index],gkprime[index]);
            //else ApplyHIOSupport(inside,gkp1[index],gkprime[index],beta_HIO);
            //else ApplyPOSHIOSupport(inside,gkp1[index],gkprime[index],beta_HIO);
	    ApplyPOSHIOSupport(inside,gkp1[index],gkprime[index],beta_HIO);
            //else {
            //ApplyDMSupport(inside,gkp1[index], gkprime[index], pmpsg[index], gammas, gammam, beta);
            //}
            //ApplyERSupport(inside,pmpsg[index],gkp1[index]);
            //ApplyHIOSupport(inside,gkp1[index],gkprime[index],beta);
            //else ApplySFSupport(inside,gkp1[index],gkprime[index]);
            epsilonS+=rms(tmp[0]-gkp1[index][0],tmp[1]-gkp1[index][1]);
	  }
        },ap);
      if(iter>=1){
        fepF<<sqrt(epsilonF/row/column)<<endl;
        fepS<<sqrt(epsilonS/row/column)<<endl;
      }
      if(iter==0) {
        convertFromFFTWToOpencv(*cache,row, column, gkp1, MOD2,0);
        imwrite("recon_support.png",*cache);
        convertFromFFTWToOpencv(*cache,row, column, gkp1, PHASE,0);
        imwrite("recon_phase_support.png",*cache);
      }

      //if(sqrt(epsilonS/row/column)<0.05) break;
      fftresult = fftw(gkp1,row,column,fftresult,1); // FFT to get f field;
      if(useDM) pmpsg = fftw(pmpsg,row,column,pmpsg,1); // FFT to get f field;
      //end iteration
    }
    fepF.close();
    fepS.close();

    convertFromFFTWToOpencv(*cache,row, column, gkp1, MOD2,0);
    imwrite("recon_intensity.png",*cache);
    convertFromFFTWToOpencv(*cache,row, column, gkp1, PHASE,0);
    imwrite("recon_phase.png",*cache);
    if(useDM)  convertFromFFTWToOpencv(*cache,row, column, pmpsg, MOD2,1);
    if(useDM)  imwrite("recon_pmpsg.png",*cache);
    convertFromFFTWToOpencv(*cache,row, column, fftresult, MOD2,1);
    imwrite("recon_pattern.png",*cache);
}

int main(int argc, char** argv )
{

    if(argc < 2){
      printf("please feed the object intensity and phase image\n");
    }
    bool runSim;
    bool simCCDbit = 0;
    if(argv[1] == std::string("sim")){
      runSim = 1;
    }else{
      runSim = 0;
    }
    auto seed = (unsigned)time(NULL);
    bool isFresnel = 0;
    bool useGaussionLumination = 0;
    bool simHERALDO = 1;
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
    fftw_complex *gkp1 = 0;
    fftw_complex *targetfft = 0;
    fftw_complex* fftresult = 0;
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
    if(simHERALDO){
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
    setups.pixelsize = setups.d/oversampling/setups.beamspotsize*setups.lambda;
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
        if(oversampling>1) gkp1 = convertFromOpencvToFFTW(*(cache = extend(*cache1,oversampling)), gkp1,0,"waveFront");
	else {
	  gkp1 = convertFromOpencvToFFTW(*(cache = cache1), gkp1,0,"waveFront");
	}
      }
      if(!isFarField && isFresnel){
        auto f = [&](int x, int y, fftw_complex &data){
          auto tmp = (complex<double>*)&data;
	  double phase = pi*setups.lambda*setups.d/pow(setups.pixelsize,2)*(pow((x-0.5*row)/row,2)+pow((y-0.5*column)/column,2))/10;
	  *tmp *= exp(complex<double>(0,phase));
	};
        imageLoop<decltype(f)>(gkp1,row,column,&f,0);
      }
      if(useGaussionLumination){
        setups.spt = &cir3;
      for(int i = 0; i<row; i++){ //apply support on O field; Ps
        for(int j = 0; j<column; j++){ 
          bool inside = cir2.isInside(i,j);
          ApplyERSupport(inside,gkp1[i*column+j],gkp1[i*column+j]);
        }
      }
      //diffraction image, either from simulation or from experiments.
        auto f = [&](int x, int y, fftw_complex &data){
          auto tmp = (complex<double>*)&data;
	  *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2);
	  //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
	};
        imageLoop<decltype(f)>(gkp1,row,column,&f,0);
      }
      convertFromFFTWToOpencv(*cache,row, column, gkp1, MOD2,0,1,"Object MOD2");
      imwrite("init_object.png",*cache);
      convertFromFFTWToOpencv(*cache,row, column, gkp1, PHASE,0,1,"Object Phase");
      imwrite("init_object_phase.png",*cache);
      targetfft = fftw(gkp1,row,column,targetfft,1); 
    }else{
      if(mergeDepth == 1) cache = cache1;
      else cache = new Mat(row, column, format_cv);
      targetfft = convertFromOpencvToFFTW(*cache1,targetfft,1); 
    }
    if(restart){
      intensity = readImage(argv[3]);
      Mat phase = readImage(argv[4]);
      gkp1 = createWaveFront(intensity, phase, row, column,cache,gkp1);
      fftresult = fftw(gkp1,row,column,fftresult,1); //If not restart, this line just allocate space, the values are not used.
    }else{
      fftresult = (fftw_complex*) malloc(sz);
    }
    //cir2.x0=row/2;
    //cir2.y0=column/2;
    double decay = 1;
    std::default_random_engine generator;
    std::poisson_distribution<int> distribution(1000);
    fftw_complex *autocorrelation = (fftw_complex*) malloc(sz);
    for(int i = 0; i<row*column; i++){ //remove the phase information
     // double randphase = arg(tmp);//static_cast<double>(rand())/RAND_MAX*2*pi;
      int tx = i/row;
      if(tx >= row/2) tx -= row/2;
      if(i/row < row/2) tx += row/2;
      int ty = i%row;
      if(ty >= column/2) ty -= column/2;
      if(i%row < column/2) ty += column/2;
      double mod = rms(targetfft[i][0],targetfft[i][1])*decay*sqrt(2);
      if(runSim&&simCCDbit) {
	int range= 65536;
        //mod = sqrt(((double)floor(pow(mod,2)*range))/(range)); //assuming we use 16bit CCD
	mod = sqrt(pow(mod,2)+double(distribution(generator))/range); //Poisson noise
      }
      if(1){
      if(setups.useBS && cir.isInside(tx,ty)) {
        targetfft[i][0] = 0;
        targetfft[i][1] = 0;
      }
      else{
        //complex<double> tmp(targetfft[i][0],targetfft[i][1]);
        double randphase = static_cast<double>(rand())/RAND_MAX*2*pi;
        targetfft[i][0] = mod*cos(randphase);
        targetfft[i][1] = mod*sin(randphase);
      }
      autocorrelation[i][0] = pow(mod,2)*(tx-row/2)*(ty-column/2)/20;
      autocorrelation[i][1] = 0;
      }
    }
    convertFromFFTWToOpencv(*cache,row, column, autocorrelation, MOD,1,1,"HERALDO U core");
    imwrite("ucore.png",*cache);
    autocorrelation = fftw(autocorrelation, row, column, autocorrelation, 0);
    if(!restart){
      memcpy(fftresult, targetfft,sz);
    }
    convertFromFFTWToOpencv(*cache,row, column, targetfft, MOD2,1,1,"Pattern MOD2");
    //imwrite("init_pattern.png",*cache);
    targetfft = convertFromOpencvToFFTW(*cache, targetfft , 1 , "waveFront");
    convertFromFFTWToOpencv(*cache,row, column, targetfft, MOD2,1,1,"Pattern MOD2");
    imwrite("init_pattern.png",*cache);
    convertFromFFTWToOpencv(*cache,row, column, autocorrelation, MOD2,1,1,"Autocorrelation MOD2");
    imwrite("auto_correlation.png",*cache);
    convertFromFFTWToOpencv(*cache,row, column, targetfft, PHASE,1);
    imwrite("init_phase.png",*cache);
    //Phase retrieving starts from here. In the following, only targetfft is needed.
    phaseRetrieve(setups, row, column, targetfft, gkp1, cache, fftresult); //fftresult is the starting point of the iteration
    return 0;
}
