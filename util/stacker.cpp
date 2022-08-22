#include <stdio.h>
#include <fftw3.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
using pixeltype=uint16_t;
auto format_cv = CV_16UC(1);
static const int rcolor = pow(2,16);
//using pixeltype=char;
//auto format_cv = CV_8UC(1);
double getRatio(Mat &bkg, Mat &lowint, Mat &highint){
  int row = lowint.rows;
  int column = lowint.cols;
  double tot1 = 0;
  double tot2 = 0;
  pixeltype* rowp;
  for(int x = 0; x < row ; x++){
    rowp = lowint.ptr<pixeltype>(x);
    for(int y = 0; y<column; y++){
      if(rowp[y]>=rcolor/2 && rowp[y]<rcolor-1){
        pixeltype &bkgval = bkg.ptr<pixeltype>(x)[y];
	pixeltype highval = highint.ptr<pixeltype>(x)[y];
        tot1+=rowp[y]-bkgval;
        tot2+=highval-bkgval;
      }
    }
  }
    printf("%f,%f,%f\n",tot1,tot2,tot1/tot2);
  return tot1/tot2;
}
int main(int argc, char** argv )
{
    if ( argc < 3 )
    {
        printf("usage: ./DisplayImage img1.png img2.png ...\n");
        printf("usage: img*.png are same patterns with different exposure, the sequence is of increasing exposure time.\n");
        printf("usage: The output is the logarithm of the intensity.\n");
        return -1;
    }
    vector<Mat> imagein;
    int mergeDepth = 8;
    Mat bkg = imread(argv[1], IMREAD_UNCHANGED);
    int row = bkg.rows;
    int column = bkg.cols;
    vector<Mat*> imageout;
    for(int i = 2 ; i < argc; i++){
      imagein.push_back( imread( argv[i], IMREAD_UNCHANGED ) );
      imageout.push_back( new Mat(row/mergeDepth, column/mergeDepth, format_cv, Scalar::all(0)));
    }
    //Mat imagein = imread( argv[1], IMREAD_GRAYSCALE );
    int threshold = 1;
//    int row = 1000;
//    int column = 1000;
    Mat image (row, column, format_cv, Scalar::all(0)); //log image
    Mat imagelogmerged (row/mergeDepth, column/mergeDepth, format_cv, Scalar::all(0)); //log image
    Mat imagefloat (row/mergeDepth, column/mergeDepth, CV_64FC1, Scalar::all(0));
    //cout << imagein<<endl;
    pixeltype* rowp;
    pixeltype* rowo;
    pixeltype* rowb;
    double* rowf;
    //char* rowo;
    int tot = 0;
    double totx = 0;
    double toty = 0;
    double sumx = 0;
    double sumy = 0;
    int max = 0;
    vector<double> ratios;
    ratios.resize(imagein.size()-1,0);
    //firstly, calculate the relative exposure time.
    double maxratio = 1;
    for(int i = 0 ; i < imagein.size()-1; i++){
      ratios[i] = getRatio(bkg,imagein[i],imagein[i+1]);
      maxratio*=ratios[i];
    }
    ratios.push_back(1);
    printf("maxratio=%f\n",maxratio);
    double maxIntensity = 0;
    for(int x = 0; x < row ; x++){
	rowp = imagein[0].ptr<pixeltype>(x);
	rowb = bkg.ptr<pixeltype>(x);
	rowo =   image.ptr<pixeltype>(x);
	rowf =   imagefloat.ptr<double>(x/mergeDepth);
        for(int y = 0; y<column; y++){
	    double intensity = rowp[y];
	    if(intensity >= rcolor-1){
              double ratio = ratios[0];
              for(int i = 1;i < imagein.size(); i++){
                intensity = imagein[i].ptr<pixeltype>(x)[y];
                if(intensity<rcolor-1){
                  intensity = ratio*(imagein[i].ptr<pixeltype>(x)[y]-rowb[y]);
                  break;
		}
                else if(i != imagein.size()-1)
                  ratio*=ratios[i];
		else{
                  intensity = ratio*(imagein[i].ptr<pixeltype>(x)[y]);
		}
	      }
	    }else
              intensity-=rowb[y];

	    rowf[y/mergeDepth]+=intensity/(rcolor-1)/maxratio/mergeDepth/mergeDepth;
	    if(maxIntensity < rowf[y/mergeDepth]) maxIntensity = rowf[y/mergeDepth];

            rowo[y]=std::max(0.,floor(log2(intensity/maxratio)*pow(2,12)));//log2(rowp[y])*pow(2,11);
	    tot+= rowp[y];
	    totx += double(rowp[y]);
	    toty += double(rowp[y]);
	    sumx += double(rowp[y])*x/row;
	    sumy += double(rowp[y])*y/row;
	    if(max < rowp[y]) max = rowp[y];
	}
	//printf("\n");
    }
    printf("\ntot=%d,max=%d,middle=(%f,%f), max=%f\n",tot,max,sumx/totx,sumy/toty,maxIntensity);
    double *data = (double*) imagefloat.data;
    for(int i = 0; i < imagefloat.total(); i++){
      double rat = 1;
      for(int j = 0; j < imageout.size(); j++){
        ((uint16_t*)imageout[j]->data)[i] = (uint16_t)floor(min(1.,std::max(0.,rat*data[i]))*65535);
        if(j<imageout.size()-1) rat*=ratios[j];
      }
      ((pixeltype*)imagelogmerged.data)[i]=std::max(0.,floor((log2(data[i])+16)*pow(2,12)));//log2(rowp[y])*pow(2,11);
      data[i]/=maxIntensity;
    }

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    //namedWindow("Display Image", WINDOW_FREERATIO );
    imwrite("logimage.png", image);
    imwrite("imageout0.png", *imageout[0]);
    //imwrite("imageout1.png", *imageout[1]);
    //imwrite("imageout2.png", *imageout[2]);
    imwrite("floatimage.tiff", imagefloat);
    imwrite("logimagemerged.png", imagelogmerged);
    return 0;
}
