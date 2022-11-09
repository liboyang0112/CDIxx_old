//
// Total Variation Filter using CUDA
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include "tvFilter.h"
#include <iostream>
#include <string>
#include <stdio.h>
#include "imageReader.h"

using namespace std;
using namespace cv;


// Program main
int main( int argc, char** argv ) {

   // name of image
   string image_name = "sample";

   // input & output file names
   string input_file =  argv[1];
   string output_file_cpu = image_name+"_cpu.png";
   string output_file_gpu = image_name+"_gpu.png";

   // Read input image 
   cv::Mat srcImage = readImage(input_file.c_str());
   Mat* floatImage;
   if(srcImage.depth()!=CV_32F) floatImage = convertFromIntegerToReal(srcImage);
   else {
     srcImage = imread( input_file, IMREAD_UNCHANGED  );
     floatImage = &srcImage;
     //auto f = [&](int x, int y, Real &data1, fftw_format &data2){
     //  data1 = data2.real();
     //};
     //imageLoop<decltype(f), Real, fftw_format>(floatImage, &srcImage, &f);
   }
   if(srcImage.empty())
   {
      std::cout<<"Image Not Found: "<< input_file << std::endl;
      return -1;
   }
   cout <<"\ninput image size: "<<srcImage.cols<<" "<<srcImage.rows<<" "<<srcImage.channels()<<"\n";

   // convert RGB to gray scale
   //cv::cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
  
   // Declare the output image  
   cv::Mat dstImage_gpu (floatImage->size(), floatImage->type());
   // run total variation filter on GPU  
   tvFilter(*floatImage, dstImage_gpu);
   // normalization to 0-255
   Mat* output = convertFromComplexToInteger<Real>(&dstImage_gpu, 0, MOD);
   imwrite(output_file_gpu, *output);


   // Declare the output image  
   //cv::Mat dstImage_cpu (srcImage.size(), srcImage.type());
   //// run total variation filter on CPU  
   //tvFilter_CPU(srcImage, dstImage_cpu);
   //// normalization to 0-255
   //dstImage_cpu.convertTo(dstImage_cpu, CV_32F, 1.0 / 255, 0);
   //dstImage_cpu*=255;
   //// Output image
   //imwrite(output_file_cpu, dstImage_cpu);
      
   return 0;
}





