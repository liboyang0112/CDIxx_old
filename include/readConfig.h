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
  bool doKCDI = 0;
  bool useDM=0;
  bool useBS=0;
  bool useShrinkMap = 0;
  bool reconAC = 0;
  bool phaseModulation = 0;
  bool restart=0;
  bool saveIter=0;
//integer configs
  int beamStopSize = 10;
  int nIter = 2000;
  int nIterKCDI = 2000;
  int noiseLevel = 0;
//float configs
  Real exposure = 0.1;
  Real exposureKCDI = 0.1;
  Real oversampling = 3;
  Real lambda = 0.01;
  Real d = 16e3;
  Real pixelsize = 26;
  Real beamspotsize = 200;
  Real shrinkThreshold = 0.15;
  std::string algorithm = "200*RAAR";
  CDIfiles common;
  CDIfiles KCDI;
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
