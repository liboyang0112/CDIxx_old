
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
// Declare the variables
Mat src, dst;
int top, bottom, left, right;
int borderType = BORDER_CONSTANT;
int main( int argc, char** argv )
{
    const char* imageName = argc >=2 ? argv[1] : "lena.jpg";
    // Loads an image
    src = imread( samples::findFile( imageName )); // Load an image
    // Check if image is loaded fine
    if( src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default lena.jpg] \n");
        return -1;
    }
    // Initialize arguments for the filter
    top = (int) (src.rows); bottom = top;
    left = (int) (src.cols); right = left;
    Scalar value(0);
    copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
    imwrite( "output.png", dst );
    return 0;
}
