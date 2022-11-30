#include "format.h"

class mnistData{
  void* dataset;
  int idat;
	public:
  mnistData(const char* dir, int &row, int &col);
  Real* read();
  ~mnistData();
};
