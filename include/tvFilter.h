#include "format.h"
void tvFilter(const cv::Mat& input, cv::Mat& output, Real noiseLevel=25, int nIters=100);
Real* tvFilterWrap(Real* d_input, Real noiseLevel, int nIters);
void inittvFilter(int row, int col);
