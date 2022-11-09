#include <common.h>

Mat readImage(const char* name, bool isFrequency = 0, Mat **mask = 0);
Mat* extend( Mat &src , Real ratio, Real val = 0);
Real getVal(mode m, fftw_format &data);
Real getVal(mode m, Real &data);
template<typename T=fftw_format>
Mat* convertFromComplexToInteger(Mat *fftwImage, Mat* opencvImage = 0, mode m = MOD, bool isFrequency = 0, Real decay = 1, const char* label= "default",bool islog = 0){
  pixeltype* rowo;
  T* rowp;
  int row = fftwImage->rows;
  int column = fftwImage->cols;
  if(!opencvImage) opencvImage = new Mat(row,column,format_cv);
  Real tot = 0;
  int tot1 = 0;
  Real max = 0;
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowo = opencvImage->ptr<pixeltype>(targetx);
    rowp = fftwImage->ptr<T>(x);
    for(int y = 0; y<column; y++){
      Real target = getVal(m, rowp[y]);
      tot += target;
      if(max < target) max = target;
      if(islog){
        if(target!=0)
          target = log2(target)*rcolor/log2(rcolor)+rcolor;
	if(target < 0) target = 0;
	
      }
      else target*=rcolor*decay;
      if(target<0) target = -target;

      if(target>=rcolor) {
	      //printf("larger than maximum of %s png %f\n",label, target);
	      target=rcolor-1;
	      //target=0;
      }
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      rowo[targety] = floor(target);
      tot1+=rowo[targety];
      //if(opencv_reverted) rowp[targety] = rcolor - 1 - rowp[targety];
      //rowp[targety] = rcolor - 1 - rowp[targety];
    }
  }
  printf("total intensity %s: raw average %4.2e, image average: %d, max: %f\n", label, tot/row/column, tot1/row/column, max);
  return opencvImage;
}

Mat* convertFromIntegerToComplex(Mat &image, Mat* cache = 0, bool isFrequency = 0, const char* label= "default");
Mat* convertFromIntegerToReal(Mat &image, Mat* cache = 0, bool isFrequency = 0, const char* label= "default");
Mat* convertFromIntegerToComplex(Mat &image,Mat &phase,Mat* cache = 0);
