#define Bits 16

static tbb::affinity_partitioner ap;
using namespace cv;
// Declare the variables
using namespace std;
static const int mergeDepth = 1; //use it only when input image is integers
#if Bits==12
using pixeltype=uint16_t;
static const int nbits = 12;
static const auto format_cv = CV_16UC1;
#elif Bits==16
using pixeltype=uint16_t;
static const int nbits = 16;
static const auto format_cv = CV_16UC1;
#else
using pixeltype=uchar;
static const int nbits = 8;
static const auto format_cv = CV_8UC1;
#endif
//using inputtype=uchar;
//static const int inputbits = 8;

static const int rcolor = pow(2,nbits);
static bool opencv_reverted = 0;
static const double scale = 1;

const double pi = 3.1415927;
