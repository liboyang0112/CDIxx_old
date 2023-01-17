#include "experimentConfig.h"
class monoChromo : public experimentConfig{
  public:
    void *locplan;
    Real *spectra;  //normalized spectra
    Real *lambdas;  //normalized lambdas, 1 is the shortest
    int nlambda;
    int *rows;
    int *cols;
    monoChromo(const char* configfile);
    void init(int nrow, int ncol, int nlambda_, Real* lambdas_, Real* spectra_);
    void init(int nrow, int ncol, Real* lambdasi, Real* spectrumi, Real endlambda);
    void generateMWL(void* d_input, void* d_patternSum, void* single = 0);
    void solveMWL(void* d_input, void* d_patternSum, void* initial = 0);
};
