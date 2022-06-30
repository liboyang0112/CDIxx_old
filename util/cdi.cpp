# include <stdlib.h>
# include <cassert>
# include <stdio.h>
# include <time.h>
# include <random>
# include <complex>

#include <stdio.h>
#include <opencv2/opencv.hpp>
# include <fftw3.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
using namespace cv;
// Declare the variables
using namespace std;
static const int nbits = 8;
static const int rcolor = pow(2,nbits);

const double pi = 3.1415927;
using namespace cv;
double gaussian(double x, double y, double sigma){
  double r2 = pow(x,2) + pow(y,2);
  return 1./sqrt(2*pi*sigma)*exp(-r2/2/pow(sigma,2));
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
  fftw_plan plan_backward;
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

fftw_complex* convertFromOpencvToFFTW(Mat I, int row, int column, fftw_complex* cache = 0){
  if(!cache) cache = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
  int tot = 0;
  uchar* rowp;
  for(int x = 0; x < row ; x++){
    rowp = I.ptr<uchar>(x);
    for(int y = 0; y<column; y++){
      tot += rowp[y];
      cache[x*column+y][0] = sqrt(((double)rowp[y])/rcolor);
      cache[x*column+y][1] = 0;
    }
  }
  printf("total intensity: %d\n", tot);
  return cache;
}

fftw_complex* convertFromOpencvToFFTW(Mat I,Mat phase, int row, int column, fftw_complex* cache = 0){
  if(!cache) cache = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * row * column );
  int tot = 0;
  uchar *rowi, *rowp;
  for(int x = 0; x < row ; x++){
    rowi = I.ptr<uchar>(x);
    rowp = phase.ptr<uchar>(x);
    for(int y = 0; y<column; y++){
      tot += rowp[y];
      double phase = rowp[y];
      phase*=0.2*pi/rcolor;
      phase-=pi;
      cache[x*column+y][0] = sqrt(((double)rowi[y])/rcolor)*cos(phase);
      cache[x*column+y][1] = sqrt(((double)rowi[y])/rcolor)*sin(phase);
    }
  }
  printf("total intensity: %d\n", tot);
  return cache;
}

void convertFromFFTWToOpencv(Mat &I, int row, int column, fftw_complex* cache, mode m, bool isFrequency, double decay = 1){
  uchar* rowp;
  int tot = 0;
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = I.ptr<uchar>(targetx);
    for(int y = 0; y<column; y++){
      double target;
      complex<double> tmpc(cache[x*column+y][0],cache[x*column+y][1]);
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
        target = (std::arg(tmpc)+pi)/2/pi;
          break;
        default:
          target = cache[x*column+y][0];
      }
      target*=rcolor*decay;
      tot += (int)target;
      if(target<0) target = -target;
      if(target>=rcolor) target=rcolor-1;
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      rowp[targety] = floor(target);
    }
  }
  printf("total intensity: %d\n",tot);
}
void applyMod(fftw_complex* source, fftw_complex* target, int row, int column){
  assert(source!=0);
  assert(target!=0);
  for(int i = 0; i < row*column ; i++){
    double mod = rms(target[i][0],target[i][1]);
    double phase;
    complex<double> tmpc(source[i][0],source[i][1]);
    phase = std::arg(tmpc);
    source[i][0] = cos(phase)*mod;
    source[i][1] = sin(phase)*mod;
  }
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
   //   printf("skipping %d, %d, %d\n",tx, ty, i);
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
fftw_complex* createWaveFront(Mat intensity, Mat phase, int &rows, int &columns, Mat* &itptr, fftw_complex* wavefront = 0){
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
  if(intensity.channels()==3){
    itptr = new Mat(rows, columns, CV_8UC1);
    cv::cvtColor(intensity, *itptr, cv::COLOR_BGR2GRAY);
  }else{
    itptr = &intensity;
  }
  Mat &intensity_sc = *itptr;
  if(phase.channels()==3){
    imageptr = new Mat(rows, columns, CV_8UC1);
    cv::cvtColor(phase, *imageptr, cv::COLOR_BGR2GRAY);
  }else{
    imageptr = &phase;
  }
  Mat &phase_sc = *imageptr;
  //wavefront = convertFromOpencvToFFTW(intensity_sc, rows, columns, wavefront);
  wavefront = convertFromOpencvToFFTW(intensity_sc, phase_sc, rows, columns, wavefront);
  delete imageptr;
  return wavefront;
  //imwrite("input.png",image);
}


Mat extend( Mat src , double ratio)
{
  Mat dst;
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
  Scalar value(0);
  copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
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
  if(insideS){
    rhonp1[0] = rhoprime[0]*( rhoprime[0] > 0 );
    rhonp1[0] = rhoprime[0];
    if(rhonp1[0]>1) rhonp1[0] = 1;
  }else{
    rhonp1[0] -= beta*rhoprime[0];
    if(rhonp1[0]<0) rhonp1[0]=0;
    if(rhonp1[0]>1) rhonp1[0]=1;
  }
  rhonp1[1] = 0;
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

  complex<double> rho(rhonp1[0],rhonp1[1]);
  complex<double> rhoprime(rhop[0],rhop[1]);
  complex<double> pmpsrho(pmsrho[0],pmsrho[1]);
  if(insideS){
    rho = (1-beta*gammam)*rhoprime+beta*(1+gammam+gammas)*rhoprime-beta*(1+gammas)*pmpsrho;
  }else{
    rho += beta*gammas*rhoprime-beta*(1+gammas)*pmpsrho;
  }
  rhonp1[0] = rho.real();
  rhonp1[1] = rho.imag();
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
int main(int argc, char** argv )
{
    if(argc < 2){
      printf("please feed the object intensity and phase image\n");
    }
    auto seed = (unsigned)time(NULL);
    srand(seed);
    printf("seed:%d\n",seed);
    Mat intensity = imread( argv[1], 1 );
    Mat phase = imread( argv[2], 1 );
    double oversampling = 3;
    fftw_complex *gkp1 = 0;
    fftw_complex *targetfft = 0;
    fftw_complex* fftresult = 0;
    fftw_complex* gkprime = 0;
    fftw_complex* pmpsg = 0;
    bool useDM = 0;
    bool restart = 0;
    bool useBS = 0; //beam stop
    if(argc > 3){
      restart = 1;
      
    }
    int row, column;
    Mat* cache;
    gkp1 = createWaveFront(extend(intensity,oversampling), extend(phase,oversampling), row, column,cache,gkp1);
    targetfft = fftw(gkp1,row,column,targetfft,1);  //The diffraction image, the starting point
    if(restart){
      intensity = imread(argv[3],1);
      phase = imread(argv[4],1);
      gkp1 = createWaveFront(intensity, phase, row, column,cache,gkp1);
    }
    fftresult = fftw(gkp1,row,column,fftresult,1); //If not restart, this line just allocate space, the values are not used.
    if(useDM) pmpsg = fftw(gkp1,row,column,pmpsg,1);//just to allocate space, the values are not used.
    C_circle cir;
    cir.x0=row/2;
    cir.y0=column/2;
    cir.r=1;
    rect re;
    re.startx = (oversampling-1)/2*row/oversampling;
    re.starty = (oversampling-1)/2*column/oversampling;
    re.endx = row-re.startx;
    re.endy = column-re.starty;
    if(1)
    for(int i = 0; i<row*column; i++){ //remove the phase information
     // double randphase = arg(tmp);//static_cast<double>(rand())/RAND_MAX*2*pi;
      int tx = i/row;
      if(tx >= row/2) tx -= row/2;
      if(i/row < row/2) tx += row/2;
      int ty = i%row;
      if(ty >= column/2) ty -= column/2;
      if(i%row < column/2) ty += column/2;
      double mod = rms(targetfft[i][0],targetfft[i][1]);
      if(useBS && cir.isInside(tx,ty)) {
        targetfft[i][0] = targetfft[i][1] = 0;
      }
      else{
        //complex<double> tmp(targetfft[i][0],targetfft[i][1]);
        double randphase = static_cast<double>(rand())/RAND_MAX*2*pi;
        targetfft[i][0] = mod*cos(randphase);
        targetfft[i][1] = mod*sin(randphase);
      }
      if(!restart){
        fftresult[i][0] = targetfft[i][0];
        fftresult[i][1] = targetfft[i][1];
      }
      if(useDM){
        pmpsg[i][0] = fftresult[i][0];
        pmpsg[i][1] = fftresult[i][1];
      }
    }
    convertFromFFTWToOpencv(*cache,row, column, targetfft, MOD2,1);
    imwrite("init_pattern.png",*cache);
    convertFromFFTWToOpencv(*cache,row, column, targetfft, PHASE,1);
    imwrite("init_phase.png",*cache);
    //Phase retrieving starts from here. In the following, only targetfft is needed.
    double beta = -1;
    double beta_HIO = 0.9;
    double gammas = -1./beta;
    double gammam = 1./beta;
    double epsilonS, epsilonF;
    gkp1 = fftw(targetfft,row,column,gkp1,0); //IFFT to get O field;
    std::ofstream fepF,fepS;
    fepF.open("epsilonF.txt",ios::out |(restart? ios::app:std::ios_base::openmode(0)));
    fepS.open("epsilonS.txt",ios::out |(restart? ios::app:std::ios_base::openmode(0)));
    int niters = 1000;
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
      if(useDM) applyMod(pmpsg,targetfft,row,column);  
      epsilonS = epsilonF = 0;
      gkprime = fftw(fftresult,row,column,gkprime,0);
      if(useDM) pmpsg = fftw(pmpsg,row,column,pmpsg,0);
      for(int i = 0; i<row; i++){ //apply support on O field; Ps
        for(int j = 0; j<column; j++){ 
          epsilonF+=rms(gkp1[i*column+j][0]-gkprime[i*column+j][0],gkp1[i*column+j][1]-gkprime[i*column+j][1]);
          fftw_complex tmp = {gkp1[i*column+j][0],gkp1[i*column+j][1]};
          bool inside = re.isInside(i,j);
          //if(iter >= niters - 20 ) ApplyERSupport(inside,gkp1[i*column+j],gkprime[i*column+j]);
          //if(iter >= niters - 20 || iter % 20 == 0) ApplyPOSERSupport(inside,gkp1[i*column+j],gkprime[i*column+j]);
          //if(iter >= niters - 20 || iter<20) ApplyERSupport(inside,gkp1[i*column+j],gkprime[i*column+j]);
          //if(iter >= niters - 20) ApplyERSupport(inside,gkp1[i*column+j],gkprime[i*column+j]);
          //ApplyERSupport(re.isInside(i,j),gkp1[i*column+j],gkprime[i*column+j]);
          //else ApplyHIOSupport(inside,gkp1[i*column+j],gkprime[i*column+j],beta_HIO);
          //else ApplyPOSHIOSupport(inside,gkp1[i*column+j],gkprime[i*column+j],beta_HIO);
	  ApplyHIOSupport(inside,gkp1[i*column+j],gkprime[i*column+j],beta_HIO);
          //else {
          //ApplyPOSDMSupport(inside,gkp1[i*column+j], gkprime[i*column+j], pmpsg[i*column+j], gammas, gammam, beta);
          //}
          //ApplyPOSERSupport(inside,pmpsg[i*column+j],gkp1[i*column+j]);
          //ApplyHIOSupport(re.isInside(i,j),gkp1[i*column+j],gkprime[i*column+j],beta);
          //else ApplySFSupport(re.isInside(i,j),gkp1[i*column+j],gkprime[i*column+j]);

          epsilonS+=rms(tmp[0]-gkp1[i*column+j][0],tmp[1]-gkp1[i*column+j][1]);
        }
      }
      if(iter>=1){
        fepF<<sqrt(epsilonF/row/column)<<endl;
        fepS<<sqrt(epsilonS/row/column)<<endl;
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
    return 0;
}
