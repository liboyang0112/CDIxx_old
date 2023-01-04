#include "mnistData.h"
#include "cudaConfig.h"
#include "common.h"

cuMnist::cuMnist(const char* dir, int re, int r, int c) : mnistData(dir, rowraw, colraw), refinement(re), row(r), col(c){
  cuRaw = memMngr.borrowCache(rowraw*colraw*sizeof(Real));
  rowrf = rowraw*refinement;
  colrf = colraw*refinement;
  cuRefine = memMngr.borrowCache(rowrf*colrf*sizeof(Real));
};

void cuMnist::cuRead(void* out){
  cudaMemcpy(cuRaw, read(), rowraw*colraw*sizeof(Real), cudaMemcpyHostToDevice);
  init_cuda_image(rowrf, colrf, 65536, 1);
  cudaF(refine)((Real*)cuRaw, (Real*)cuRefine, refinement);
  init_cuda_image(row, col, 65536, 1);
  cudaF(pad)((Real*)cuRefine, (Real*)out, rowrf, colrf);
}
