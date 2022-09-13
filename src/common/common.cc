#include "common.h"
#include "fftw.h"
Mat* convertFromIntegerToComplex(Mat &image, Mat* cache, bool isFrequency, const char* label){
  int row = image.rows;
  int column = image.cols;
  if(!cache) cache = new Mat(row, column, CV_64FC2);
  double tot = 0;
  pixeltype* rowp;
  fftw_complex* rowo;
  int targetx, targety;
  for(int x = 0; x < row ; x++){
    if(isFrequency){
      targetx = x<row/2?x+row/2:(x-row/2);
    }else{
      targetx = x;
    }
    rowp = image.ptr<pixeltype>(x);
    rowo = cache->ptr<fftw_complex>(targetx);
    for(int y = 0; y<column; y++){
      if(isFrequency){
        targety = y<column/2?y+column/2:(y-column/2);
      }else{
	targety = y;
      }
      double intensity = ((double)rowp[y])/(rcolor-1);
      fftw_complex &datatmp = rowo[targety];
      if(opencv_reverted) intensity = 1-intensity;
      datatmp[0] = sqrt(intensity);
      datatmp[1] = 0;
      tot += sqrt(intensity);
    }
  }
  printf("total intensity %s: %f\n",label, tot);
  return cache;
}

Mat* convertFromIntegerToComplex(Mat &image,Mat &phase,Mat* cache){
  int row = image.rows;
  int column = image.cols;
  if(!cache) cache = new Mat(row, column, CV_64FC2);
  int tot = 0;
  pixeltype *rowi, *rowp;
  fftw_complex *rowo;
  for(int x = 0; x < row ; x++){
    rowi = image.ptr<pixeltype>(x);
    rowp = phase.ptr<pixeltype>(x);
    rowo = phase.ptr<fftw_complex>(x);
    for(int y = 0; y<column; y++){
      tot += rowp[y];
      double phase = rowp[y];
      //phase*=2*pi/rcolor;
      //phase-=pi;
      phase = static_cast<double>(rand())/RAND_MAX*2*pi;
      rowo[y][0] = sqrt(((double)rowi[y])/rcolor)*cos(phase);
      rowo[y][1] = sqrt(((double)rowi[y])/rcolor)*sin(phase);
    }
  }
  printf("total intensity: %d\n", tot);
  return cache;
}
double getVal(mode m, fftw_complex &data){
  complex<double> &tmpc = *(complex<double>*)(data);
  switch(m){
    case MOD:
      return std::abs(tmpc);
      break;
    case MOD2:
      return pow(std::abs(tmpc),2);
      break;
    case IMAG:
      return tmpc.imag();
      break;
    case PHASE:
      if(std::abs(tmpc)==0) return 0;
      return (std::arg(tmpc)+pi)/2/pi;
      break;
    default:
      return tmpc.real();
  }
}
double getVal(mode m, double &data){
  return data;
}

Mat* multiWLGen(Mat* original, Mat* output, double m, int step, double dphilambda, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	if(!output) output = new Mat(original->rows/m,original->cols/m, CV_64FC2);
	Mat *merged = new Mat(original->rows/m,original->cols/m, CV_64FC2);
	int max = original->rows*(1-1./m)/2;
	double weight = double(step)/max;
	for(int im = 0; im < max; im+=step){ // im = 0, m*lambda, im=max lambda.
		startx = starty = im;
		resize((*original)(Range(startx, original->rows-startx),Range(starty, original->cols-starty)), *merged,merged->size());
		double scale = double(output->rows)/(original->rows-2*im);
		if(dphilambda!=0){
			double dphi = 2*scale*dphilambda;
        		auto f = [&](int x, int y, complex<double> &data){ data = abs(data)*exp(complex<double>(0,dphi)); };
        		imageLoop<decltype(f), complex<double>>(merged, &f);
		}
		merged = fftw(merged,merged,1);
		if(im == 0) {
			merged->copyTo(*output);
			normalize(*output,*output,weight*scale);
		}
		else addWeighted(*output,1,*merged,weight*scale,0,*output);
	}
	delete merged;
	return output;
}

Mat* multiWLGenAVG(Mat* original, Mat* output, double m, int step, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	if(!output) output = new Mat(original->rows/m,original->cols/m, CV_16UC1);
	Mat* mergedf = fftw(original,0,1);
	int max = original->rows*(1-1./m)/2;
	double weight = double(step)/max;
        auto f = [&](int x, int y, complex<double> &data){ data = abs(data); };
        imageLoop<decltype(f), complex<double>>(mergedf, &f);
        auto f1 = [&](int x, int y, complex<double> &data){ data = pow(abs(data),2); };
	for(int im = 0; im < max; im+=step){ // im = 0, m*lambda, im=max lambda.
		startx = starty = im;
		Mat *mergedtmp = new Mat();
		Mat *mergedt = convertFO<complex<double>>(mergedf);
		resize((*mergedt)(Range(startx, original->rows-startx),Range(starty, original->cols-starty)), *mergedtmp,output->size());
        	imageLoop<decltype(f), complex<double>>(mergedtmp, &f1);
		if(im == max/2) {
			Mat *fftwtmp = fftw(mergedtmp, 0, 1);
			Mat * tmpwrite = convertFromComplexToInteger(fftwtmp, 0 , MOD2, 1,1, "", 1);
			imwrite("cropmid.png",*tmpwrite);
			delete tmpwrite, fftwtmp;
		}
		double scale = double(mergedtmp->rows)/(original->rows-2*im);
		if(im == 0) {
			mergedtmp->copyTo(*output);
			normalize(*output,*output,weight*scale);
		}
		else addWeighted(*output,1,*mergedtmp,weight*scale,0,*output);
		delete mergedtmp;
		delete mergedt;
	}
	delete mergedf;
	return output;
}
void addMatComponent_pattern(Mat *matrix, Size sz, int x, int y, double scale){
	double newx = (x-sz.height/2) * scale + sz.height/2;
	double newy = (y-sz.width/2) * scale + sz.width/2;
	int idx = x*sz.width + y;
	if(newx <= -scale/2-0.5 || newx >= sz.height+scale/2+0.5 || newy <= -scale/2-0.5 || newy >= sz.width+scale/2+0.5) return;
	double density = 1./scale*scale;
	int startx = floor(newx-scale/2); //for these pixels, matrix[i+j*row][x+y*row] += density;
	int starty = floor(newy-scale/2);
	int endx = ceil(newx+scale/2);
	int endy = ceil(newy+scale/2);
	for(int i = max(0,startx); i <= min(sz.height-1,endx); i++){
		double wtx = 1;
		if(i == startx) wtx*=newx-scale/2-startx;
		if(i == endx) wtx*=endx-newx-scale/2;
		for(int j = max(0,starty); j <= min(sz.width-1,endy); j++){
			double weight = wtx;
			if(j == starty) weight*=newy-scale/2-starty;
			if(j == endy) weight*=endy-newy-scale/2;
			printf("(%d,%d),[%d,%d]+=%f\n",i,j,i*sz.width+j,idx,density*weight);
			//matrix->ptr<T>(i*sz.width+j)[idx] += density*weight;
		}
	}
}
void addMatComponent(Mat *matrix, Size sz, int x, int y, double scale){
	double newx = (x-sz.height/2) * scale + sz.height/2;
	double newy = (y-sz.width/2) * scale + sz.width/2;
	int idx = x*sz.width + y;
	if(newx <= -scale/2-0.5 || newx >= sz.height+scale/2+0.5 || newy <= -scale/2-0.5 || newy >= sz.width+scale/2+0.5) return;
	double density = 1./scale*scale;
	int startx = floor(newx-scale/2); //for these pixels, matrix[i+j*row][x+y*row] += density;
	int starty = floor(newy-scale/2);
	int endx = ceil(newx+scale/2);
	int endy = ceil(newy+scale/2);
	for(int i = max(0,startx); i <= min(sz.height-1,endx); i++){
		double wtx = 1;
		if(i == startx) wtx*=newx-scale/2-startx;
		if(i == endx) wtx*=endx-newx-scale/2;
		for(int j = max(0,starty); j <= min(sz.width-1,endy); j++){
			double weight = wtx;
			if(j == starty) weight*=newy-scale/2-starty;
			if(j == endy) weight*=endy-newy-scale/2;
			printf("(%d,%d),[%d,%d]+=%f\n",i,j,i*sz.width+j,idx,density*weight);
			//matrix->ptr<T>(i*sz.width+j)[idx] += density*weight;
		}
	}
}
Mat* multiWLGenAVG_MAT(Mat* original, Mat* output, double m, int step, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	if(!output) output = new Mat(original->rows,original->cols, CV_16UC1);
	Mat* mergedf = fftw(original,0,1);
	int max = original->rows*(1-1./m)/2;
	double weight = double(step)/max;
        auto f = [&](int x, int y, complex<double> &data){ data = abs(data); };
        imageLoop<decltype(f), complex<double>>(mergedf, &f);
        auto f1 = [&](int x, int y, complex<double> &data){ data = pow(abs(data),2); };
	Mat *matrix = new Mat(original->total(),original->total(),CV_16UC1);
	addMatComponent(matrix, original->size(), original->rows/2, original->cols/2, m);
	exit(0);
	delete mergedf;
	return output;
}
void plotColor(const char* name, Mat* logged){
	Mat dst8 = Mat::zeros(logged->size(), CV_8U);
	normalize(*logged, *logged, 0, 255, NORM_MINMAX);
	convertScaleAbs(*logged, dst8);
	applyColorMap(dst8, dst8, COLORMAP_TURBO);
	imwrite(name,dst8);
}
