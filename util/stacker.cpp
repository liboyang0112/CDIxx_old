#include <stdio.h>
#include <fftw3.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "format.h"
#include "imageReader.h"
using namespace cv;
using namespace std;
//using pixeltype=char;
//auto format_cv = CV_8UC(1);

Mat centerCrop(Mat input, Real centxr, Real centyr){
  int rows = input.rows;
  int cols = input.cols;
  int centx = centxr*rows;
  int centy = centyr*cols;
  int outrow = (rows-abs(centx-rows/2)*2)/16*16;
  int outcol = (cols-abs(centy-cols/2)*2)/16*16;
  outrow = outcol = min(outrow, outcol);
  Mat output(outrow, outcol, float_cv_format(1));
  Real* inputrow;
  Real* outputrow;
  for(int x = 0; x < outrow; x++){
    inputrow = input.ptr<Real>(centx-outrow/2+x);
    outputrow = output.ptr<Real>(x);
    for(int y = 0; y < outcol; y++)
      outputrow[y] = inputrow[centy-outcol/2+y];
  }
  return output;
}

Real getRatio(Mat &bkg, Mat &lowint, Mat &highint){
  int row = lowint.rows;
  int column = lowint.cols;
  Real tot1 = 0;
  Real tot2 = 0;
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
  int mergeDepth = 4;
  vector<Mat*> imageout;
  for(int i = 2 ; i < argc; i++){
    imagein.push_back( imread( argv[i], IMREAD_UNCHANGED ) );
  }
  int row = imagein[0].rows;
  int column = imagein[0].cols;
  Mat *bkgptr;
  try{
    bkgptr = new Mat(row, column, format_cv, Scalar::all(stoi(argv[1])));
  }catch(std::invalid_argument&){
    bkgptr = new Mat(imread(argv[1], IMREAD_UNCHANGED));
  }
  Mat &bkg = *bkgptr;
  //Mat imagein = imread( argv[1], IMREAD_GRAYSCALE );
  int threshold = 1;
  //    int row = 1000;
  //    int column = 1000;
  Mat image (row, column, format_cv, Scalar::all(0)); //log image
  Mat imagefloatin (row, column, float_cv_format(1), Scalar::all(0));
  //cout << imagein<<endl;
  pixeltype* rowp;
  pixeltype* rowo;
  pixeltype* rowb;
  Real* rowf;
  Real* rowfi;
  char* rowoe;
  //char* rowo;
  int tot = 0;
  Real totx = 0;
  Real toty = 0;
  Real sumx = 0;
  Real sumy = 0;
  int max = 0;
  vector<Real> ratios;
  ratios.resize(imagein.size()-1,0);
  //firstly, calculate the relative exposure time.
  Real maxratio = 1;
  for(int i = 0 ; i < imagein.size()-1; i++){
    ratios[i] = getRatio(bkg,imagein[i],imagein[i+1]);
    maxratio*=ratios[i];
  }
  ratios.push_back(1);
  printf("maxratio=%f\n",maxratio);
  Real maxIntensity = 0;
  for(int x = 0; x < row ; x++){
    rowp = imagein[0].ptr<pixeltype>(x);
    rowb = bkg.ptr<pixeltype>(x);
    rowo =   image.ptr<pixeltype>(x);
    rowfi =   imagefloatin.ptr<Real>(x);
    for(int y = 0; y<column; y++){
      Real intensity = rowp[y];
      if(intensity >= rcolor-1 && imagein.size()!=1){
        Real ratio = ratios[0];
        for(int i = 1;i < imagein.size(); i++){
          intensity = imagein[i].ptr<pixeltype>(x)[y];
          if(intensity<rcolor-1 || i == imagein.size()-1){
            intensity = ratio*(intensity-rowb[y]); //to avoid negative value of type uint16_t
            break;
          }
          ratio*=ratios[i];
        }
      }else
        intensity-=rowb[y];
      rowfi[y] = intensity/(rcolor-1)/maxratio;

      rowo[y]=std::max(0.,floor(log2(intensity/maxratio)*pow(2,12)));//log2(rowp[y])*pow(2,11);
      if(maxIntensity < rowfi[y]) maxIntensity = rowfi[y];
      tot+= rowp[y];
      if(rowfi[y] >= 0.4){
      totx += Real(rowp[y]);
      toty += Real(rowp[y]);
      sumx += Real(rowp[y])*x/row;
      sumy += Real(rowp[y])*y/row;
      }
      if(max < rowp[y]) max = rowp[y];
    }
    //printf("\n");
  }
  printf("\ntot=%d,max=%d,middle=(%f,%f), max=%f\n",tot,max,sumx/=totx,sumy/=toty,maxIntensity);
  Mat imagefloatc = centerCrop(imagefloatin, sumx, sumy);
  row = imagefloatc.rows;
  column = imagefloatc.cols;
  for(int i = 0; i < imagein.size(); i++)
    imageout.push_back( new Mat(row/mergeDepth, column/mergeDepth, format_cv, Scalar::all(0)));

  Mat imagelogmerged (row/mergeDepth, column/mergeDepth, format_cv, Scalar::all(0)); //log image
  Mat imagefloat (row/mergeDepth, column/mergeDepth, float_cv_format(1), Scalar::all(0));
  Mat overexposed (row/mergeDepth, column/mergeDepth, CV_8UC1, Scalar::all(0));

  for(int x = 0; x < row ; x++){
    rowoe =   overexposed.ptr<char>(x/mergeDepth);
    rowf =   imagefloat.ptr<Real>(x/mergeDepth);
    rowfi =   imagefloatc.ptr<Real>(x);
    for(int y = 0; y<column; y++){
      if(imagein[imagein.size()-1].ptr<pixeltype>(x)[y] == rcolor-1) rowoe[y/mergeDepth] = 255;
      else rowoe[y/mergeDepth] = 0;
      rowf[y/mergeDepth]+=rowfi[y]/mergeDepth/mergeDepth/maxIntensity;
    }
  }
  Real *data = (Real*) imagefloat.data;
  for(int i = 0; i < imagefloat.total(); i++){
    Real rat = 1;
    for(int j = 0; j < imageout.size(); j++){
      ((uint16_t*)imageout[j]->data)[i] = (uint16_t)floor(min(Real(1),std::max(Real(0),rat*data[i]))*65535);
      if(j<imageout.size()-1) rat*=ratios[j];
    }
    ((pixeltype*)imagelogmerged.data)[i]=std::min(std::max(0.,floor((log2(data[i])+16)*pow(2,12))),rcolor-1.);//log2(rowp[y])*pow(2,11);
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
  plotColor("logimagemerged.png", &imagelogmerged);
  imwrite("overexposed.png", overexposed);
  return 0;
}
