#include "opencv2/opencv.hpp"

void fftw_init();
cv::Mat* fftw(cv::Mat* in, cv::Mat* out, bool isforward);
