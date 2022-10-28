#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//using pixeltype=uint16_t;
//auto format_cv = CV_16UC(1);
using pixeltype=double;
auto format_cv = CV_64FC(1);
//using pixeltype=char;
//auto format_cv = CV_8UC(1);
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
    Mat imagein = imread( argv[1], IMREAD_UNCHANGED );
    //Mat imagein = imread( argv[1], IMREAD_GRAYSCALE );
//    int row = imagein.rows;
//    int column = imagein.cols;
    int threshold = 1;
    int row = 150;
    int column = 150;
//    Mat image (row, column, format_cv, Scalar::all(0));
    cout << imagein<<endl;
    /*j
    pixeltype* rowp;
    pixeltype* rowo;
    //char* rowo;
    double tot = 0;
    double totx = 0;
    double toty = 0;
    double sumx = 0;
    double sumy = 0;
    double max = 0;
    for(int x = 0; x < row ; x++){
	rowp = imagein.ptr<pixeltype>(x);
	rowo =   image.ptr<pixeltype>(x);
	//rowo = image.ptr<char>(x);
        for(int y = 0; y<column; y++){
            //rowp[y] = (int)(255*gaussian(((double)x)/row-0.5,((double)y)/column-0.5,0.25));
	    //if(rowp[y]>0) rowo[y]=rowp[y]/256;//log2(rowp[y])*pow(2,11);
	    //rowo[y] = 255-rowp[y];
        //    int n = rowp[(y)];
        //    int nm1 = rowp[(y-1)];
        //    int np1 = rowp[(y+1)];
        //    if(n!=0) {
        //      int score0 = 4;
        //      if(y==0 || nm1<=threshold) score0--;
        //      if(y==column-1 || np1<=threshold) score0--;
        //      if(x==0 || imagein.ptr<char>(x-1)[y]<=threshold) score0--;
        //      if(x==row-1 || imagein.ptr<char>(x+1)[y]<=threshold) score0--;
        //      if(score0 <= 1 && rowo[y]<=threshold)
        //        rowo[y]=0;
        //      else
           //     rowo[y]=floor(log2(rowp[y])*pow(2,12));//log2(rowp[y])*pow(2,11);
                //rowo[y]=n;//log2(rowp[y])*pow(2,11);
              //printf("%d, ",n);
              printf("%f\n",rowp[y]);
	    //}
	    tot+= rowp[y];
	    totx += double(rowp[y]);
	    toty += double(rowp[y]);
	    sumx += double(rowp[y])*x/row;
	    sumy += double(rowp[y])*y/row;
	    if(max < rowp[y]) max = rowp[y];
	}
	//printf("\n");
    }
    printf("\ntot=%f,max=%f,middle=(%f,%f)\n",tot,max,sumx/totx,sumy/toty);


    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    //namedWindow("Display Image", WINDOW_FREERATIO );
    */
//    imwrite("image.png", image);
    return 0;
}
