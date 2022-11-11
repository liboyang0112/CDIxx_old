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
   int row, col;
   Real* image = readImage(input_file.c_str(), row, col);
   cout <<"\ninput image size: "<<col<<" "<<row<<" "<< 1 <<"\n";
   init_cuda_image(row,col,rcolor);
	 inittvFilter(row, col);
   tvFilter(image, 25, 100);

   Mat *imagem = new Mat(row, col, float_cv_format(1));
   Real *tmp = (Real*)imagem->data;
   imagem->data = (uchar*)image;

   
   // normalization to 0-255
   Mat* output = convertFromRealToInteger(imagem, 0, MOD);
   imagem->data = (uchar*)tmp;
   delete image;
   delete imagem;
   imwrite(output_file_gpu, *output);
      
   return 0;
}





