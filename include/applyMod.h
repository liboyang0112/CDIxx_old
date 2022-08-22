#include <cuda.h>
#include <cuda_runtime.h>

__global__ void applyMod(cufftDoubleComplex* source, cufftDoubleComplex* target, int row, int column, support *bs = 0);
