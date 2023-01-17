#include "imageReader.h"
#include "matrixGen.h"
#include "sparse.h"
#include "fftw.h"
#include "readCXI.h"
#include "common.h"
using namespace std;
Mat* extend( Mat &src , Real ratio, Real val)
{
  Mat *dst = new Mat();
  int top, bottom, left, right;
  int borderType = BORDER_CONSTANT;
  if( src.empty()) {
      printf(" Error opening image\n");
      printf(" Program Arguments: [image_name -- default lena.jpg] \n");
      exit(0);
  }
  // Initialize arguments for the filter
  top = (int) ((ratio-1)/2*src.rows); bottom = top;
  left = (int) ((ratio-1)/2*src.cols); right = left;
  Scalar value(opencv_reverted?rcolor:0);
  copyMakeBorder( src, *dst, top, bottom, left, right, borderType, value );
//  imwrite("ext.png",*dst);
  return dst;
}
Mat* convertFromIntegerToComplex(Mat &image, Mat* cache, bool isFrequency, const char* label){
  int row = image.rows;
  int column = image.cols;
  if(!cache) cache = new Mat(row, column, float_cv_format(2));
  Real tot = 0;
  pixeltype* rowp;
  fftw_format* rowo;
  int targetx, targety;
  for(int x = 0; x < row ; x++){
    if(isFrequency){
      targetx = x<row/2?x+row/2:(x-row/2);
    }else{
      targetx = x;
    }
    rowp = image.ptr<pixeltype>(x);
    rowo = cache->ptr<fftw_format>(targetx);
    for(int y = 0; y<column; y++){
      if(isFrequency){
        targety = y<column/2?y+column/2:(y-column/2);
      }else{
	targety = y;
      }
      Real intensity = ((Real)rowp[y])/(rcolor-1);
      fftw_format &datatmp = rowo[targety];
      if(opencv_reverted) intensity = 1-intensity;
      datatmp = fftw_format(sqrt(intensity),0);
      tot += sqrt(intensity);
    }
  }
  printf("total intensity %s: %f\n",label, tot);
  return cache;
}
Mat* convertFromIntegerToComplex(Mat &image,Mat &phase,Mat* cache){
  int row = image.rows;
  int column = image.cols;
  if(!cache) cache = new Mat(row, column, float_cv_format(2));
  int tot = 0;
  pixeltype *rowi, *rowp;
  fftw_format *rowo;
  for(int x = 0; x < row ; x++){
    rowi = image.ptr<pixeltype>(x);
    rowp = phase.ptr<pixeltype>(x);
    rowo = cache->ptr<fftw_format>(x);
    for(int y = 0; y<column; y++){
      tot += rowp[y];
      Real phase = rowp[y];
      phase*=2*M_PI/rcolor;
      phase-=M_PI;
      phase*=0.2;
      rowo[y] = fftw_format(sqrt(((Real)rowi[y])/rcolor)*cos(phase),sqrt(((Real)rowi[y])/rcolor)*sin(phase));
    }
  }
  printf("total intensity: %d\n", tot);
  return cache;
}
Real getVal(mode m, fftw_format &data){
  switch(m){
    case MOD:
      return std::abs(data);
      break;
    case MOD2:
      return pow(std::abs(data),2);
      break;
    case IMAG:
      return data.imag();
      break;
    case PHASE:
      if(std::abs(data)==0) return 0;
      return (std::arg(data)+M_PI)/2/M_PI;
      break;
    default:
      return data.real();
  }
}
Real getVal(mode m, Real &data){
  return data;
}

Mat* convertFromIntegerToReal(Mat &image, Mat* cache, bool isFrequency, const char* label){
  int row = image.rows;
  int column = image.cols;
  if(!cache) cache = new Mat(row, column, float_cv_format(1));
  Real tot = 0;
  pixeltype* rowp;
  Real* rowo;
  int targetx, targety;
  for(int x = 0; x < row ; x++){
    if(isFrequency){
      targetx = x<row/2?x+row/2:(x-row/2);
    }else{
      targetx = x;
    }
    rowp = image.ptr<pixeltype>(x);
    rowo = cache->ptr<Real>(targetx);
    for(int y = 0; y<column; y++){
      if(isFrequency){
        targety = y<column/2?y+column/2:(y-column/2);
      }else{
	targety = y;
      }
      Real intensity = ((Real)rowp[y])/(rcolor-1);
      if(opencv_reverted) intensity = 1-intensity;
      rowo[targety] = sqrt(intensity);
      tot += sqrt(intensity);
    }
  }
  printf("total intensity %s: %f\n",label, tot);
  return cache;
}

Mat* multiWLGen(Mat* original, Mat* output, Real m, Real step, Real dphilambda, Real *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	if(!output) output = new Mat(original->rows/m,original->cols/m, float_cv_format(2));
	Mat *merged = new Mat(original->rows/m,original->cols/m, float_cv_format(2));
	int max = original->rows*(1-1./m)/2;
	Real weight = step/max;
	for(int im = 0; im < max; im+=step){ // im = 0, m*lambda, im=max lambda.
		startx = starty = im;
		resize((*original)(Range(startx, original->rows-startx),Range(starty, original->cols-starty)), *merged,merged->size());
		Real scale = Real(output->rows)/(original->rows-2*im);
		if(dphilambda!=0){
			Real dphi = 2*scale*dphilambda;
        		auto f = [&](int x, int y, complex<Real> &data){ data = abs(data)*exp(complex<Real>(0,dphi)); };
        		imageLoop<decltype(f), complex<Real>>(merged, &f);
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

Mat* multiWLGenAVG(Mat* original, Mat* output, Real m, Real step, Real *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	if(!output) output = new Mat(original->rows/m,original->cols/m, CV_16UC1);
	Mat* mergedf = fftw(original,0,1);
	int max = original->rows*(1-1./m)/2;
	Real weight = step/max;
        auto f = [&](int x, int y, complex<Real> &data){ data = pow(abs(data),2); };
        //imageLoop<decltype(f), complex<Real>>(mergedf, &f);
        //auto f1 = [&](int x, int y, complex<Real> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<Real>>(mergedf, &f);
	Mat *autocorrelation = fftw(mergedf,0,0);
	Mat *mergedt = convertFO<complex<Real>>(autocorrelation);
	Mat *singletmp = 0;
	for(int im = 0; im < max; im+=step){ // im = 0, m*lambda, im=max lambda.
		startx = starty = im;
		Mat *mergedtmp = new Mat();
		resize((*mergedt)(Range(startx, original->rows-startx),Range(starty, original->cols-starty)), *mergedtmp,output->size());
		if(im == max/2) {
			singletmp = fftw(mergedtmp,0,1);
			Mat * tmpwrite = convertFromComplexToInteger(singletmp, 0 , MOD, 1,1, "", 0);
			imwrite("cropmid.png",*tmpwrite);
			delete tmpwrite, singletmp;
		}
		Real scale = Real(mergedtmp->rows)/(original->rows-2*im);
		if(im == 0) {
			mergedtmp->copyTo(*output);
			normalize(*output,*output,weight*scale);
		}
		else addWeighted(*output,1,*mergedtmp,weight*scale,0,*output);
		delete mergedtmp;
	}
	delete mergedf,autocorrelation,mergedt;
	output = fftw(output,output,1);
	return output;
}
void addMatComponent_area(Sparse *matrix, Size sz, int x, int y, Real scale, Real midx, Real midy, Real weighttot){
	Real newx = (x-midx) * scale + midx;
	Real newy = (y-midy) * scale + midy;
	//printf("(%d,%d),[%f,%f],%f\n",x,y,newx,newy,scale);
	int idx = x*sz.width + y;
	Real halfwidth = scale/2;
	Real hfm5 = halfwidth-0.5;
	Real hfp5 = halfwidth+0.5;
	if(newx <= -hfp5 || newx >= sz.height+hfp5|| newy <= -hfp5 || newy >= sz.width+hfp5) return;
	Real density = 1./scale/scale;
	int startx = floor(newx-hfm5); //for these pixels, matrix[i+j*row][x+y*row] += density;
	int starty = floor(newy-hfm5);
	int endx = ceil(newx+hfm5);
	int endy = ceil(newy+hfm5);
	int cord[] = {0,idx};
	for(int i = max(0,startx); i <= min(sz.height-1,endx); i++){
		Real wtx = 1;
		if(startx!=endx){
			if(i == startx) wtx=startx-newx+hfp5;
			if(i == endx) wtx=newx+hfp5-endx;
		}else{
			wtx = scale;
		}
		for(int j = max(0,starty); j <= min(sz.width-1,endy); j++){
			cord[0] = i*sz.width+j;
			Real weight = wtx;
			if(starty!=endy){
				if(j == starty) weight*=starty-newy+hfp5;
				if(j == endy) weight*=newy+hfp5-endy;
			}else{
				weight*=scale;
			}
			(*matrix)[cord] += density*weight*weighttot;
			//printf("(%d, %d)+=%f*%f=%f\n",i,j,density,weight,density*weight);
		}
	}
}
void addMatComponent_Interpolate(Sparse *matrix, Size sz, int x, int y, Real scale, Real midx, Real midy, Real weighttot){
	Real newx = (x-midx) * scale + midx;
	Real newy = (y-midy) * scale + midy;
	//printf("(%d,%d),[%f,%f],%f\n",x,y,newx,newy,scale);
	int idx = x*sz.width + y;
	if(newx <= -1 || newx >= sz.height|| newy <= -1 || newy >= sz.width) return;
	Real density = 1./scale/scale;
	int startx = floor(newx); //for these pixels, matrix[i+j*row][x+y*row] += density;
	int starty = floor(newy);
	Real dx = newx-startx;
	Real dy = newy-starty;
	int cord[] = {idx,startx*sz.width + starty};
	Real tmp = (1-dx)*(1-dy)*density;
	if(tmp!=0) (*matrix)[cord] += tmp;
	cord[1] += 1;
	tmp = (1-dx)*dy*density;
	if(tmp!=0) (*matrix)[cord] += tmp;
	cord[1] += sz.width;
	tmp = density*dx*dy;
	if(tmp!=0) (*matrix)[cord] += tmp;
	cord[1] -= 1;
	tmp = density*dx*(1-dy);
	if(tmp!=0) (*matrix)[cord] += tmp;
}
Mat* multiWLGenAVG_MAT(Mat* original, Mat* output, Real m, Real step, Real *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	Mat* mergedf = fftw(original,0,1);
	Real weight = 1;//step/(1-1./m);
        auto f = [&](int x, int y, complex<Real> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<Real>>(mergedf, &f);
	output = convertFO<complex<Real>>(mergedf);

	Real rl = 2;
	Mat* tmp = convertFromComplexToInteger(output, 0, MOD, 0, 1./4, "merged", 1);
	plotColor("originalPattern.png",tmp);
	//build C matrix
	Sparse *matrix = new Sparse(original->total(),original->total());
	printf("rl=%f, max=%f, step=%f\n",1.,m,step);
	//for(Real rl = 0.3; rl > 1./m-0.01; rl -= step){
	for(int idx = 0; idx < original->total(); idx++){
	//int idx = original->cols*300+300;
		addMatComponent_area(matrix, original->size(), idx/original->cols, idx%original->cols, rl, Real(original->rows)/2-0.5, Real(original->cols)/2-0.5,weight);
		//addMatComponent_Interpolate(matrix, original->size(), idx/original->cols, idx%original->cols, rl, Real(original->rows)/2, Real(original->cols)/2,weight);
	}
		printf("size = %lu, ratio=%f\n",matrix->matrix.size(),rl);
	//}
	//multiply the matrix
	complex<Real>* y = (*matrix)*(complex<Real>*)(output->data);
	memcpy(output->data, y, sizeof(complex<Real>)*output->total());
	delete y;
	delete mergedf;
	return output;
}
Mat* multiWLGenAVG_MAT_AC(Mat* original, Mat* output, Real m, Real step, Real *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	Mat* mergedf = fftw(original,0,1);
	if(output) delete output;
	Real weight = 1./16;//step/(1-1./m);
        auto f = [&](int x, int y, complex<Real> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<Real>>(mergedf, &f);
	Mat *autocorrelation = fftw(mergedf,0,0);
	Mat *mergedt = convertFO<complex<Real>>(autocorrelation);
	//build C matrix
	Sparse *matrix = new Sparse(original->total(),original->total());
	printf("rl=%f, max=%f, step=%f\n",1.,m,step);
	//for(Real rl = 0.3; rl > 1./m-0.01; rl -= step){
	Real rl = 0.5;
	for(int idx = 0; idx < original->total(); idx++){
	//int idx = original->cols*305+305;
		addMatComponent_area(matrix, original->size(), idx/original->cols, idx%original->cols, rl, Real(original->rows)/2, Real(original->cols)/2,weight);
		//addMatComponent_Interpolate(matrix, original->size(), idx/original->cols, idx%original->cols, rl, Real(original->rows)/2, Real(original->cols)/2,weight);
	}
		printf("size = %lu, ratio=%f\n",matrix->matrix.size(),rl);
	//}
	complex<Real>* y = (*matrix)*(complex<Real>*)(mergedt->data);
	memcpy(mergedf->data, y, sizeof(complex<Real>)*mergedf->total());
	autocorrelation = fftw(mergedf,autocorrelation,1);
	output = convertFO<complex<Real>>(autocorrelation);
	delete autocorrelation;
	delete y;
	delete mergedf;
	delete mergedt;
	return output;
}
Mat* multiWLGenAVG_MAT_FFT(Mat* original, Mat* output, Real m, Real step, Real *spectrum){ //original image, ratio between long lambda and short lambda.
	output = fftw(original,output,1);
	Real weight = 1./16;//step/(1-1./m);
        auto f = [&](int x, int y, complex<Real> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<Real>>(output, &f);
	Mat *mergedt = convertFO<complex<Real>>(output);
	//build C matrix
	Sparse *matrix = new Sparse(original->total(),original->total());
	printf("rl=%f, max=%f, step=%f\n",1.,m,step);
	//for(Real rl = 0.3; rl > 1./m-0.01; rl -= step){
	int padding = 1;
	matrixGen(matrix, original->rows, original->cols, padding, padding);
	printf("size = %lu, padding=%d\n",matrix->matrix.size(),padding);
	complex<Real>* y = (*matrix)*((complex<Real>*)(mergedt->data));
	memcpy(mergedt->data, y, sizeof(complex<Real>)*mergedt->total());
	convertFO<complex<Real>>(mergedt,output);
	delete y;
	delete mergedt;
	return output;
}
Mat* multiWLGenAVG_MAT_AC_FFT(Mat* original, Mat* output, Real m, Real step, Real *spectrum){ //original image, ratio between long lambda and short lambda.
	Mat* mergedf = fftw(original,0,1);
	if(output) delete output;
	Real weight = 1./16;//step/(1-1./m);
        auto f = [&](int x, int y, complex<Real> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<Real>>(mergedf, &f);
	Mat *autocorrelation = fftw(mergedf,0,0);
	Mat *mergedt = convertFO<complex<Real>>(autocorrelation);
	//Mat *mergedt = new Mat(original->rows, original->cols, float_cv_format(2), Scalar(0));
	//mergedt->ptr<fftw_format>(299-160)[299-160] = 1;
	//mergedt->ptr<fftw_format>(161)[161] = 1;
	//build C matrix
	Sparse *matrix = new Sparse(original->total(),original->total());
	printf("rl=%f, max=%f, step=%f\n",1.,m,step);
	double rl = 1.37;
	double pix = original->rows*(rl-1)/2;
	//for(Real rl = 0.3; rl > 1./m-0.01; rl -= step){
	//matrixGen(matrix, original->rows, original->cols, 0, 0, 1);
	//matrixGen(matrix, original->rows, original->cols, 50, 50);
	matrixGen(matrix, original->rows, original->cols, pix, pix,1./pow(rl,2));
	//matrixGen(matrix, original->rows, original->cols, 150, 150, 1./4);
	//printf("size = %lu, padding=%d\n",matrix->matrix.size(),padding);
	complex<Real>* y = matrix->TMultiply((complex<Real>*)(mergedt->data));
	memcpy(mergedf->data, y, sizeof(complex<Real>)*mergedf->total());
	autocorrelation = fftw(mergedf,autocorrelation,1);
	output = convertFO<complex<Real>>(autocorrelation);
	delete autocorrelation;
	delete y;
	delete mergedf;
	delete mergedt;
	return output;
}
Mat* multiWLGenAVG_AC_FFT(Mat* original, Mat* output, Real m, Real step, Real *spectrum){ //original image, ratio between long lambda and short lambda.
	Mat* mergedf = fftw(original,0,1);
	if(output) delete output;
	Real weight = 1./16;//step/(1-1./m);
        auto f = [&](int x, int y, complex<Real> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<Real>>(mergedf, &f);
	Mat *autocorrelation = fftw(mergedf,0,0);
	//Mat *mergedt = new Mat(original->rows, original->cols, float_cv_format(2), Scalar(0));
	//mergedt->ptr<fftw_format>(299-160)[299-160] = 1;
	//mergedt->ptr<fftw_format>(160)[160] = 1;
	Mat *mergedt = convertFO<complex<Real>>(autocorrelation);
	delete autocorrelation;
	double r = 1.37;
	autocorrelation = extend(*mergedt,r);
	fftw(autocorrelation,autocorrelation,1);
	Mat *pattern = convertFO<fftw_format>(autocorrelation);
	int startx = original->rows*(r-1)/2;
	int starty = original->cols*(r-1)/2;
	Mat *tmp = new Mat((*pattern)(Range(startx, pattern->rows-startx),Range(starty, pattern->cols-starty)));
	output = new Mat();
	tmp->copyTo(*output);
	delete tmp;
	delete autocorrelation;
	autocorrelation = fftw(output,0,0);
	tmp = convertFromComplexToInteger(autocorrelation, 0, REAL, 0, 1, "debug", 0);
	imwrite("debug.png",*tmp);
	delete mergedf;
	delete mergedt;
	return output;
}
void plotColor(const char* name, Mat* logged){
	Mat dst8 = Mat::zeros(logged->size(), CV_8U);
	normalize(*logged, *logged, 0, 255, NORM_MINMAX);
	convertScaleAbs(*logged, dst8);
	applyColorMap(dst8, dst8, COLORMAP_TURBO);
	imwrite(name,dst8);
}
