#include <stdio.h>
#include <fftw3-mpi.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "format.h"
using namespace cv;
using namespace std;
using pixeltype=uint16_t;
auto format_cv = CV_16UC(1);
static const int rcolor = pow(2,16);
//using pixeltype=char;
//auto format_cv = CV_8UC(1);
double getRatio(Mat &lowint, Mat &highint){
  int row = lowint.rows;
  int column = lowint.cols;
  double tot1 = 0;
  double tot2 = 0;
  pixeltype* rowp;
  for(int x = 0; x < row ; x++){
    rowp = lowint.ptr<pixeltype>(x);
    for(int y = 0; y<column; y++){
      if(rowp[y]>=rcolor/2 && rowp[y]<rcolor-1){
	pixeltype highval = highint.ptr<pixeltype>(x)[y];
        tot1+=rowp[y];
        tot2+=highval;
      }
    }
  }
    printf("%f,%f,%f\n",tot1,tot2,tot1/tot2);
  return tot1/tot2;
}
int main(int argc, char** argv )
{
	const int row = 100;
	const int column = 100;
	Mat image(row, column, float_cv_format(2));
	if ( !image.data )
	{
	    printf("No image data \n");
	    return -1;
	}
	printf("channels=%d\n",image.flags);
	for(double i = 0; i < row ; i++){
	    	fftw_complex* rowp = image.ptr<fftw_complex>(i);
	    	for(double j = 0; j < column ; j++){
			if(i!=0||j!=0) {
				rowp[int(j)][0] = i/hypot(i,j),j/hypot(i,j);
				rowp[int(j)][1] = j/hypot(i,j);
			}
		}
	}
	for(int i = 0; i < row*column ; i++){
		cout<<((complex<double>*)image.data)[i]<<endl;
	}
    
    //namedWindow("Display Image", WINDOW_FREERATIO );
    //imwrite("logimage.tiff", image);
    return 0;
}
