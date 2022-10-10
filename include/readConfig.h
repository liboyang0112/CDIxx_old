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
#include "format.h"

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.

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
  bool runSim;
  bool simCCDbit = 1;
  bool isFresnel = 1;
  bool doIteration = 1;
  bool useGaussionLumination = 0;
  bool useGaussionHERALDO = 0;
  bool doCentral =0;
  bool useRectHERALDO = 0;
  bool doKCDI = 1;
  bool useDM=0;
  bool useBS=0;
  bool useShrinkMap = 0;
  bool reconAC = 0;
  bool phaseModulation = 0;
  bool restart=0;
  bool saveIter=0;
//float configs
  Real exposure = 0.1;
  Real oversampling = 3;
  Real lambda = 0.01;
  Real d = 16e3;
  Real pixelsize = 26;
  Real beamspotsize = 200;
  CDIfiles common;
  CDIfiles KCDI;
  readConfig(const char* configfile);
  void print();
  ~readConfig(){};
};
#endif
