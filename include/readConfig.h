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

class readConfig
{
public:
//bool configs
  bool runSim = 0;
  bool simCCDbit = 0;
  bool isFresnel = 0;
  bool doIteration = 0;
  bool useGaussionLumination = 0;
  bool useGaussionHERALDO = 0;
  bool doCentral =0;
  bool useRectHERALDO = 0;
  bool dopupil = 0;
  bool useDM=0;
  bool useBS=0;
  bool useShrinkMap = 0;
  bool reconAC = 0;
  bool phaseModulation = 0;
  bool restart=0;
  bool saveIter=0;
  bool domnist = 0;
//integer configs
  int beamStopSize = 10;
  int nIter = 2000;
  int nIterpupil = 2000;
  int noiseLevel = 0;
  int noiseLevel_pupil = 0;
  int verbose = 0;
  int mnistN = 3000;
//float configs
  Real exposure = 0.1;
  Real exposurepupil = 0.1;
  Real oversampling = 3;
  Real oversampling_spt = 2;
  Real lambda = 0.01;
  Real d = 16e3;
  Real dpupil = 100.;
  Real pixelsize = 26;
  Real beamspotsize = 200;
  Real shrinkThreshold = 0.15;
  Real positionUncertainty = 3;
  Real costheta = 1.; // 1 means normal incidence.
  std::string mnistData = "data";
  std::string algorithm = "200*RAAR";
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
