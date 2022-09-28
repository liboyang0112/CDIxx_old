#include "imageReader.h"
#include "readCXI.h"
#include "common.h"
using namespace std;
template <typename inputtype>
Mat read16bitImage(Mat imagein, int nbitsimg)
{
    int row = imagein.rows;
    int column = imagein.cols;
    //int threshold = 1;
    int factor = pow(2,16-nbitsimg);
    Mat image(row/mergeDepth, column/mergeDepth, CV_16UC(1), Scalar::all(0));
    inputtype* rowp;
    uint16_t* rowo;
    int tot = 0;
    int max = 0;
    for(int x = 0; x < row ; x++){
	rowp = imagein.ptr<inputtype>(x);
	rowo = image.ptr<uint16_t>(x/mergeDepth);
        for(int y = 0; y<column; y++){
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
	    rowo[y/mergeDepth] += rowp[y]*factor/mergeDepth/mergeDepth;
	    tot+= rowp[y];
	    if(max < rowp[y]) max = rowp[y];
	}
    }
    printf("\ntot=%d,max=%d\n",tot,max);
    imwrite("input.png",image);
    return image;
}

Mat readImage(char* name, bool isFrequency, Mat **mask){
  if(string(name).find(".cxi")!=string::npos){
    printf("Input is recognized as cxi file\n");
    Mat imagein = readCXI(name, mask);  //32FC2, max 65535
    Mat image(imagein.rows, imagein.cols, float_cv_format(2));
    auto f = [&](int x, int y, complex<float> &data, fftw_format &dataout){
      dataout = fftw_format(sqrt(max(Real(0),data.real()/rcolor)),0);
    };
    imageLoop<decltype(f),complex<float>,fftw_format>(&imagein,&image,&f,1);
    return image;
  }
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
    printf("input image nbits: 8, channels=%d",imagein.channels());
    if(imagein.channels()>=3){
      Mat image(imagein.rows, imagein.cols, CV_8UC1);
      cv::cvtColor(imagein, image, cv::COLOR_BGR2GRAY);
      return read16bitImage<char>(image,8);
    }else
      return read16bitImage<char>(imagein,8);
  }else if(imagein.depth() == CV_16U){
    printf("input image nbits: 16");
    return read16bitImage<uint16_t>(imagein,16);
  }else{  //Image data is float
    printf("Image depth %d is not recognized as integer type (%d or %d), Image data is treated as floats\n", imagein.depth(), CV_8U, CV_16U);
    Mat *tmp = convertFromComplexToInteger<Real>(&imagein,0,MOD,0,1,"input",1); //Here we save the logarithm of the input image
    imwrite("inputs.png", *tmp);
    delete tmp;
    Mat image(imagein.rows, imagein.cols, float_cv_format(2));
    auto f = [&](int x, int y, Real &data, fftw_format &dataout){
      dataout = fftw_format(sqrt(max(Real(0),data)),0);
    };
    imageLoop<decltype(f),Real,fftw_format>(&imagein,&image,&f,1);
    return image;
  }
}

Mat* extend( Mat &src , Real ratio, Real val)
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
//  imwrite("ext.png",*dst);
  return dst;
}
