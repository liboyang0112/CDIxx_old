#include "common.h"
#include "matrixGen.h"
#include "sparse.h"
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

Mat* multiWLGen(Mat* original, Mat* output, double m, double step, double dphilambda, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	if(!output) output = new Mat(original->rows/m,original->cols/m, CV_64FC2);
	Mat *merged = new Mat(original->rows/m,original->cols/m, CV_64FC2);
	int max = original->rows*(1-1./m)/2;
	double weight = step/max;
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

Mat* multiWLGenAVG(Mat* original, Mat* output, double m, double step, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	if(!output) output = new Mat(original->rows/m,original->cols/m, CV_16UC1);
	Mat* mergedf = fftw(original,0,1);
	int max = original->rows*(1-1./m)/2;
	double weight = step/max;
        auto f = [&](int x, int y, complex<double> &data){ data = pow(abs(data),2); };
        //imageLoop<decltype(f), complex<double>>(mergedf, &f);
        //auto f1 = [&](int x, int y, complex<double> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<double>>(mergedf, &f);
	Mat *autocorrelation = fftw(mergedf,0,0);
	Mat *mergedt = convertFO<complex<double>>(autocorrelation);
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
		double scale = double(mergedtmp->rows)/(original->rows-2*im);
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
void addMatComponent_area(Sparse *matrix, Size sz, int x, int y, double scale, double midx, double midy, double weighttot){
	double newx = (x-midx) * scale + midx;
	double newy = (y-midy) * scale + midy;
	//printf("(%d,%d),[%f,%f],%f\n",x,y,newx,newy,scale);
	int idx = x*sz.width + y;
	double halfwidth = scale/2;
	double hfm5 = halfwidth-0.5;
	double hfp5 = halfwidth+0.5;
	if(newx <= -hfp5 || newx >= sz.height+hfp5|| newy <= -hfp5 || newy >= sz.width+hfp5) return;
	double density = 1./scale/scale;
	int startx = floor(newx-hfm5); //for these pixels, matrix[i+j*row][x+y*row] += density;
	int starty = floor(newy-hfm5);
	int endx = ceil(newx+hfm5);
	int endy = ceil(newy+hfm5);
	int cord[] = {0,idx};
	for(int i = max(0,startx); i <= min(sz.height-1,endx); i++){
		double wtx = 1;
		if(startx!=endx){
			if(i == startx) wtx=startx-newx+hfp5;
			if(i == endx) wtx=newx+hfp5-endx;
		}else{
			wtx = scale;
		}
		for(int j = max(0,starty); j <= min(sz.width-1,endy); j++){
			cord[0] = i*sz.width+j;
			double weight = wtx;
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
void addMatComponent_Interpolate(Sparse *matrix, Size sz, int x, int y, double scale, double midx, double midy, double weighttot){
	double newx = (x-midx) * scale + midx;
	double newy = (y-midy) * scale + midy;
	//printf("(%d,%d),[%f,%f],%f\n",x,y,newx,newy,scale);
	int idx = x*sz.width + y;
	if(newx <= -1 || newx >= sz.height|| newy <= -1 || newy >= sz.width) return;
	double density = 1./scale/scale;
	int startx = floor(newx); //for these pixels, matrix[i+j*row][x+y*row] += density;
	int starty = floor(newy);
	double dx = newx-startx;
	double dy = newy-starty;
	int cord[] = {idx,startx*sz.width + starty};
	double tmp = (1-dx)*(1-dy)*density;
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
Mat* multiWLGenAVG_MAT(Mat* original, Mat* output, double m, double step, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	Mat* mergedf = fftw(original,0,1);
	double weight = 1./pow(2,2);//step/(1-1./m);
        auto f = [&](int x, int y, complex<double> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<double>>(mergedf, &f);
	output = convertFO<complex<double>>(mergedf);
	//build C matrix
	Sparse *matrix = new Sparse(original->total(),original->total());
	printf("rl=%f, max=%f, step=%f\n",1.,m,step);
	//for(double rl = 0.3; rl > 1./m-0.01; rl -= step){
	double rl = 2;
	for(int idx = 0; idx < original->total(); idx++){
	//int idx = original->cols*300+300;
		addMatComponent_area(matrix, original->size(), idx/original->cols, idx%original->cols, rl, double(original->rows)/2-0.5, double(original->cols)/2-0.5,weight);
		//addMatComponent_Interpolate(matrix, original->size(), idx/original->cols, idx%original->cols, rl, double(original->rows)/2, double(original->cols)/2,weight);
	}
		printf("size = %lu, ratio=%f\n",matrix->matrix.size(),rl);
	//}
	//multiply the matrix
	complex<double>* y = (*matrix)*(complex<double>*)(output->data);
	memcpy(output->data, y, sizeof(complex<double>)*output->total());
	delete y;
	delete mergedf;
	return output;
}
Mat* multiWLGenAVG_MAT_AC(Mat* original, Mat* output, double m, double step, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	Mat* mergedf = fftw(original,0,1);
	if(output) delete output;
	double weight = 1./16;//step/(1-1./m);
        auto f = [&](int x, int y, complex<double> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<double>>(mergedf, &f);
	Mat *autocorrelation = fftw(mergedf,0,0);
	Mat *mergedt = convertFO<complex<double>>(autocorrelation);
	//build C matrix
	Sparse *matrix = new Sparse(original->total(),original->total());
	printf("rl=%f, max=%f, step=%f\n",1.,m,step);
	//for(double rl = 0.3; rl > 1./m-0.01; rl -= step){
	double rl = 0.5;
	for(int idx = 0; idx < original->total(); idx++){
	//int idx = original->cols*305+305;
		addMatComponent_area(matrix, original->size(), idx/original->cols, idx%original->cols, rl, double(original->rows)/2, double(original->cols)/2,weight);
		//addMatComponent_Interpolate(matrix, original->size(), idx/original->cols, idx%original->cols, rl, double(original->rows)/2, double(original->cols)/2,weight);
	}
		printf("size = %lu, ratio=%f\n",matrix->matrix.size(),rl);
	//}
	complex<double>* y = (*matrix)*(complex<double>*)(mergedt->data);
	memcpy(mergedf->data, y, sizeof(complex<double>)*mergedf->total());
	autocorrelation = fftw(mergedf,autocorrelation,1);
	output = convertFO<complex<double>>(autocorrelation);
	delete autocorrelation;
	delete y;
	delete mergedf;
	delete mergedt;
	return output;
}
Mat* multiWLGenAVG_MAT_FFT(Mat* original, Mat* output, double m, double step, double *spectrum){ //original image, ratio between long lambda and short lambda.
	int startx = 0;
	int starty = 0;
	Mat* mergedf = fftw(original,0,1);
	if(output) delete output;
	double weight = 1./16;//step/(1-1./m);
        auto f = [&](int x, int y, complex<double> &data){ data = pow(abs(data),2); };
        imageLoop<decltype(f), complex<double>>(mergedf, &f);
	Mat *autocorrelation = fftw(mergedf,0,0);
	Mat *mergedt = convertFO<complex<double>>(autocorrelation);
	//build C matrix
	Sparse *matrix = new Sparse(original->total(),original->total());
	printf("rl=%f, max=%f, step=%f\n",1.,m,step);
	//for(double rl = 0.3; rl > 1./m-0.01; rl -= step){
	int padding = 2;
	matrixGen(matrix, original->rows, original->cols, padding, padding);
	printf("size = %lu, padding=%d\n",matrix->matrix.size(),padding);
	complex<double>* y = (*matrix)*(complex<double>*)(mergedt->data);
	memcpy(mergedf->data, y, sizeof(complex<double>)*mergedf->total());
	autocorrelation = fftw(mergedf,autocorrelation,1);
	output = convertFO<complex<double>>(autocorrelation);
	delete autocorrelation;
	delete y;
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
