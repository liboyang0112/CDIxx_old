#include "format.h"

class mnistData{
  void* dataset;
  int idat;
	public:
  mnistData(const char* dir, int &row, int &col);
  Real* read();
  ~mnistData();
};

class cuMnist : mnistData{
  void *cuOut, *cuRaw, *cuRefine;
  int rowraw;
  int colraw;
  int rowrf;
  int colrf;
  int refinement;
  int row;
  int col;
  public:
  cuMnist(const char* dir, int re, int r, int c);
  void cuRead(void*);
};
