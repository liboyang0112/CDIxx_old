#include <iostream>
#include <vector>
#include "cudaConfig.h"
#include "cuPlotter.h"
#include "mnistData.h"
#include "common.h"


int main(int argc, char** argv){
  if(argc==1) { printf("Tell me which one is the mnist data folder\n"); }
	int row, col, rowraw, colraw, rowrf, colrf;
	mnistData dat(argv[1],rowraw, colraw);
	int refinement = 3;
	rowrf = rowraw*refinement;
	colrf = colraw*refinement;
	col = row = 128;
	Real exposure = 0.5;
	Real minos = 2;
	Real os[] = {minos,minos*11/9,minos*11/7};
	Real maxos = os[2];
	Real ratio[] = {0.3,0.4,0.3};
	Real* d_input;
	Real* d_inputraw;
	Real* d_inputrf;
	complexFormat *d_intensity;
	complexFormat *d_patternAmp;
	Real *d_pattern, *d_patternSum;
	cudaMalloc((void**)&d_input, row*col*sizeof(Real));
	cudaMalloc((void**)&d_inputrf, rowrf*colrf*sizeof(Real));
	cudaMalloc((void**)&d_inputraw, rowraw*colraw*sizeof(Real));
	cudaMalloc((void**)&d_intensity, row*col*sizeof(Real)*maxos*maxos*2);
	cudaMalloc((void**)&d_pattern, row*col*sizeof(Real)*minos*minos);
	cudaMalloc((void**)&d_patternAmp, row*col*sizeof(Real)*minos*minos*2);
	cudaMalloc((void**)&d_patternSum, row*col*sizeof(Real)*minos*minos);
	cuPlotter plt;
	plt.init(row*minos, col*minos);

	for(int j = 0; j < 3000; j++){
		cudaMemcpy(d_inputraw, dat.read(), rowraw*colraw*sizeof(Real), cudaMemcpyHostToDevice);
		init_cuda_image(rowrf, colrf, 65536, 1);
		cudaF(refine)(d_inputraw, d_inputrf, refinement);
		init_cuda_image(row, col, 65536, 1);
		cudaF(pad)(d_inputrf, d_input, rowrf, colrf);
		for(int i = 0; i < 3; i++){
			int thisrow = row*os[i];
			int thiscol = col*os[i];
			init_cuda_image(thisrow, thiscol, 65536, 1);
			cudaF(createWaveFront)(d_input, 0, d_intensity, os[i]);
			if(i==0) plt.plotComplex(d_intensity, MOD2, 0, 1, ("object"+to_string(j)).c_str(), 0);
			myCufftExec( *plan, d_intensity,d_intensity, CUFFT_FORWARD);
			cudaF(cudaConvertFO)(d_intensity);
			init_cuda_image(row*minos, col*minos, 65536, 1);
			cudaF(crop)(d_intensity,d_patternAmp,thisrow,thiscol);
			cudaF(applyNorm)(d_patternAmp, exposure*ratio[i]/sqrt(thiscol*thisrow));
			if(i==0) {
				cudaF(getMod2)(d_patternSum, d_patternAmp);
				plt.plotFloat(d_patternSum, MOD, 0, 1./ratio[i], ("single"+to_string(j)).c_str(), 0);
				//plt.plotFloat(d_patternSum, MOD, 0, 1./ratio[i], ("singlelog"+to_string(i)).c_str(), 1);
			}else{
				cudaF(getMod2)(d_pattern, d_patternAmp);
				cudaF(add)(d_patternSum, d_pattern, 1);
			}
		}
		plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
		//plt.plotFloat(d_patternSum, MOD, 0, 1, "mergedlog", 1);
	}

	return 0;
}

