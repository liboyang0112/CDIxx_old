#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
double gaussian(double x, double y, double sigma){
    double r2 = pow(x,2) + pow(y,2);
    return exp(-r2/pow(sigma,2));
}
int main(int argc, char** argv )
{
//   if ( argc != 2 )
//   {
//       printf("usage: ./DisplayImage <Image_Path>\n");
//       return -1;
//   }
//    Mat image;
//    image = imread( argv[1], 1 );
    int row = 1000;
    int column = 1000;
    Mat image (row, column, CV_8UC(1), Scalar::all(0));
    uchar* rowp;
    for(int x = 0; x < row ; x++){
	rowp = image.ptr<uchar>(x);
        for(int y = 0; y<column; y++){
            rowp[y] = (int)(255*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.25));
	}
    }


    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_FREERATIO );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}
