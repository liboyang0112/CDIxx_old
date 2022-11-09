//
// CUDA implementation of Total Variation Filter
// Implementation of Nonlinear total variation based noise removal algorithms : 10.1016/0167-2789(92)90242-F
//
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cudaConfig.h"
#include "common.h"

#define BLOCK_SIZE      16
#define FILTER_WIDTH    1      //total width=2n+1 
#define FILTER_HEIGHT   1       

using namespace std;

// Run Total Variation Filter on GPU

#include <cub/device/device_reduce.cuh>
#include <curand_kernel.h>
struct CustomSum
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a+b;
    }
};

Real *d_output, *d_bracket, *d_lambdacore, *lambda;
static int rows, cols;
size_t sz;
CustomSum sum_op;

const short tilewidth=BLOCK_SIZE+2*FILTER_HEIGHT;



template <typename T>
__device__ T minmod(T data1, T data2){
  if(data1*data2<=0) return T(0);  
  if(data1<0) return max(data1,data2);
  return min(data1,data2);
}
template <typename T, typename Trand>
__global__ void addnoise(T *srcImage, int noiseLevel, Trand *state)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  curand_init(1,index,0,&state[index]);
  srcImage[index]+=(T(curand_poisson(&state[index], noiseLevel))-noiseLevel)/cuda_rcolor;
}
template <typename T>
__global__ void calcBracketLambda(T *srcImage, T *bracket, T* u0, T* lambdacore, T noiseLevel)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  float dt = 1e-7*noiseLevel;
  float sigmafactor = cuda_rcolor*1e-7*cuda_rcolor/(cuda_row*cuda_column*2);
  extern __shared__ float tile[];
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y]=(x>=FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y]=(x<cuda_row-FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x>=FILTER_WIDTH && y<cuda_column-FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x<cuda_row-FILTER_WIDTH && y<cuda_column-FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  __syncthreads();
  int centerIdx = (threadIdx.x+FILTER_WIDTH)*(tilewidth) + threadIdx.y+FILTER_HEIGHT;
  float dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  float dpyU = tile[centerIdx+1]-tile[centerIdx];
  float dmxU = tile[centerIdx]-tile[centerIdx-tilewidth];
  float dmyU = tile[centerIdx]-tile[centerIdx-1];
  float sbracket = 0;
  float denom = sqrt(pow(dpxU,2)+pow(dpyU,2));
  if(denom && x<cuda_row-1 && y<cuda_column-1) lambdacore[index] = ((u0[index+cuda_column]*dpxU+u0[index+1]*dpyU-u0[index]*(dpxU+dpyU))/denom-denom)*sigmafactor;
  else lambdacore[index] = 0;
  denom = sqrt(pow(dpxU,2)+pow(minmod(dpyU,dmyU),2));
  if(denom!=0) sbracket += dpxU/denom;
  denom = sqrt(pow(dpyU,2)+pow(minmod(dpxU,dmxU),2));
  if(denom!=0) sbracket += dpyU/denom;
  centerIdx-=1;
  dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  dpyU = tile[centerIdx+1]-tile[centerIdx];
  dmxU = tile[centerIdx]-tile[centerIdx-tilewidth];
  denom = sqrt(pow(dpyU,2)+pow(minmod(dpxU,dmxU),2));
  if(denom != 0) sbracket -= dpyU/denom;
  centerIdx-=tilewidth-1;
  dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  dpyU = tile[centerIdx+1]-tile[centerIdx];
  dmyU = tile[centerIdx]-tile[centerIdx-1];
  denom = sqrt(pow(dpxU,2)+pow(minmod(dpyU,dmyU),2));
  if(denom != 0) sbracket -= dpxU/denom;
  bracket[index] = dt*sbracket;
}

template <typename T>
__global__ void tvFilter(T *srcImage, T *bracket, T* u0, T* slambda)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  srcImage[index]+=bracket[index]-(*slambda)*(srcImage[index]-u0[index]);
}

template <typename T>
__global__ void gsFilter(T *srcImage, T *dstImage)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x >= cuda_row || y >= cuda_column) return;
  short tilewidth=BLOCK_SIZE+2*FILTER_HEIGHT;
  extern __shared__ float tile[];
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y]=(x>=FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y]=(x<cuda_row+FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x>=FILTER_WIDTH && y<cuda_column+FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x<cuda_row+FILTER_WIDTH && y<cuda_column+FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  __syncthreads();
  // only threads inside image will write results
  // Loop inside the filter to average pixel values
  float center = tile[(threadIdx.x+FILTER_WIDTH)*(tilewidth) + threadIdx.y+FILTER_HEIGHT];
  int pos = (threadIdx.x)*(tilewidth)+threadIdx.y;
  int restart = FILTER_WIDTH<<1+1;
  float sod = 0;
  for(int ky=-FILTER_HEIGHT; ky<=FILTER_HEIGHT; ky++) {
     for(int kx=-FILTER_WIDTH; kx<=FILTER_WIDTH; kx++) {
        sod += tile[pos++];
     }
     pos+=tilewidth-restart;
  }
  sod-=(1+FILTER_HEIGHT*2)*(1+FILTER_HEIGHT*2)*center;
  dstImage[x*cuda_column+y] = sod>0?sod:0;
}

// The wrapper is used to call total variation filter 
        void            *d_temp_storage = NULL;
        size_t          temp_storage_bytes = 0;
void inittvFilter(int row, int col){
  rows = row;
  cols = col;
  int rcolor = 65535;
  init_cuda_image(row,col,rcolor);
  sz = rows * cols * sizeof(Real);
  // Allocate device memory
  cudaMalloc(&d_output,sz);
  cudaMalloc(&d_bracket,sz);
  cudaMalloc(&d_lambdacore,sz);
  cudaMalloc(&lambda,sizeof(Real));
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_lambdacore, lambda, rows*cols, sum_op, 0));
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
}
Real* tvFilterWrap(Real* d_input, Real noiseLevel, int nIters){
        gpuErrchk(cudaMemcpy(d_output,d_input,sz,cudaMemcpyDeviceToDevice));

	//curandStateMRG32k3a *devstates;
	//cudaMalloc((void **)&devstates, cols * rows *
        //          sizeof(curandStateMRG32k3a));
	//addnoise<<<numBlocks,threadsPerBlock>>>(d_output, (int)noiseLevel, devstates);
	for(int i = 0; i<nIters; i++){
          calcBracketLambda<<<numBlocks,threadsPerBlock,sizeof(float)*(tilewidth)*(tilewidth)>>>(d_output, d_bracket, d_input, d_lambdacore, noiseLevel);
          gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, sz, d_lambdacore, lambda, rows*cols, sum_op, 0));
          tvFilter<<<numBlocks,threadsPerBlock>>>(d_output, d_bracket, d_input, lambda);
	}
	return d_output;
}
void tvFilter(const cv::Mat& input, cv::Mat& output, Real noiseLevel, int nIters)
{
	if(rows==0) inittvFilter(input.rows,input.cols);
	Real* d_input;
	gpuErrchk(cudaMalloc(&d_input,sz));
        cudaMemcpy(d_input,input.data,sz,cudaMemcpyHostToDevice);
	Real *d_output = tvFilterWrap(d_input, noiseLevel, nIters);
        cudaMemcpy(output.data,d_output,sz,cudaMemcpyDeviceToHost);
}
/*
void CImageObj::Total_Variation(int iter, double dt, double epsilon, double lambda)
{
	int i, j;
	int nx = m_width, ny = m_height;
	double ep2 = epsilon * epsilon;
 
	double** I_t = NewDoubleMatrix(nx, ny);
	double** I_tmp = NewDoubleMatrix(nx, ny);
	for (i = 0; i < ny; i++)
		for (j = 0; j < nx; j++)
			I_t[i][j] = I_tmp[i][j] = (double)m_imgData[i][j];
 
	for (int t = 0; t < iter; t++)
	{
		for (i = 0; i < ny; i++)
		{
			for (j = 0; j < nx; j++)
			{
				int iUp = i - 1, iDown = i + 1;
				int jLeft = j - 1, jRight = j + 1;    // 边界处理
				if (0 == i) iUp = i; if (ny - 1 == i) iDown = i;
				if (0 == j) jLeft = j; if (nx - 1 == j) jRight = j;
 
				double tmp_x = (I_t[i][jRight] - I_t[i][jLeft]) / 2.0;
				double tmp_y = (I_t[iDown][j] - I_t[iUp][j]) / 2.0;
				double tmp_xx = I_t[i][jRight] + I_t[i][jLeft] - 2 * I_t[i][j];
				double tmp_yy = I_t[iDown][j] + I_t[iUp][j] - 2 * I_t[i][j];
				double tmp_xy = (I_t[iDown][jRight] + I_t[iUp][jLeft] - I_t[iUp][jRight] - I_t[iDown][jLeft]) / 4.0;
				double tmp_num = tmp_yy * (tmp_x * tmp_x + ep2) + tmp_xx * (tmp_y * tmp_y + ep2) - 2 * tmp_x * tmp_y * tmp_xy;
				double tmp_den = pow(tmp_x * tmp_x + tmp_y * tmp_y + ep2, 1.5);
 
				I_tmp[i][j] += dt*(tmp_num / tmp_den + lambda*(m_imgData[i][j] - I_t[i][j]));
			}
		}  // 一次迭代
 
		for (i = 0; i < ny; i++)
			for (j = 0; j < nx; j++)
			{
				I_t[i][j] = I_tmp[i][j];
			}
 
	} // 迭代结束
 
	// 给图像赋值
	for (i = 0; i < ny; i++)
		for (j = 0; j < nx; j++)
		{
			double tmp = I_t[i][j];
			tmp = max(0, min(tmp, 255));
			m_imgData[i][j] = (Real)tmp;
		}
 
	DeleteDoubleMatrix(I_t, nx, ny);
	DeleteDoubleMatrix(I_tmp, nx, ny);
}
*/


