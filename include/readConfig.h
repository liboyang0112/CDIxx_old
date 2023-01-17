/**
 * @file 	readConfig.h
 * @brief A config file header.
 * @author Boyang Li
 */
/**
 * @brief 	CDIxxinXsys Library.
 */

#ifndef __CDIxxCONFIG
#define __CDIxxCONFIG
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <libconfig.h++>
#include <vector>
#include "format.h"

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
enum Algorithm {RAAR, ER, HIO};

struct CDIfiles{
  std::string Pattern;
  std::string Intensity;
  std::string Phase;
  std::string restart;
};

#define DECINT(x,y) int x = y;
#define DECBOOL(x,y) bool x = y;
#define DECREAL(x,y) Real x = y;
#define DECSTR(x,y) std::string x = y;

#define INTVAR(F) \
F(beamStopSize,5)F(nIter,1)F(nIterpupil,1)F(noiseLevel,0)F(noiseLevel_pupil,0)F(verbose,2)F(mnistN,3)

#define BOOLVAR(F) \
F(runSim,0)F(simCCDbit,0)F(isFresnel,0)F(doIteration,0)F(useGaussionLumination,0)F(useGaussionHERALDO,0)F(doCentral,0)F(useRectHERALDO,0)F(dopupil,0)F(useDM,0)F(useBS,0)F(useShrinkMap,0)F(reconAC,0)F(phaseModulation,0)F(restart,0)F(saveIter,0)F(domnist,0)

#define REALVAR(F) \
F(exposure,0)F(exposurepupil,0)F(oversampling,0)F(oversampling_spt,0)F(lambda,0)F(d,0)F(dpupil,0)F(pixelsize,0)F(beamspotsize,0)F(shrinkThreshold,0)F(positionUncertainty,0)F(costheta,1)

#define STRVAR(F) \
F(mnistData,"data")F(algorithm,"RAAR")
class readConfig
{
public:
  BOOLVAR(DECBOOL)
  INTVAR(DECINT)
  REALVAR(DECREAL)
  STRVAR(DECSTR)
//bool configs
  CDIfiles pupil;
  CDIfiles common;
  readConfig(const char* configfile);
  void print();
  ~readConfig(){};
};

class AlgoParser{
public:
  std::vector<AlgoParser*> subParsers;
  std::vector<int> count;
  std::vector<int> algoList;
  int nAlgo = 3;
  int currentAlgo;
  int currentCount;
  AlgoParser(std::string formula);
  void restart();
  int next();
};
#endif
