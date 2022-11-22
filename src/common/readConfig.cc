#include "readConfig.h"
#include  <memory>
#include <vector>
#include <regex>
#include <string>
#include <iostream>
using namespace std;
using namespace libconfig;

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.

int readConfigFile(const char * filename, Config &cfg)
{

  // Read the file. If there is an error, report it and exit.
  try
  {
    cfg.readFile(filename);
  }
  catch(const FileIOException &fioex)
  {
    std::cerr << "I/O error while reading file." << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const ParseException &pex)
  {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    return(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}

readConfig::readConfig(const char* configfile){
  libconfig::Config cfg;
  int ret = readConfigFile(configfile, cfg);
  if(ret==EXIT_FAILURE) exit(ret);

  // Get the store name.
  try
  {
    string name = cfg.lookup("name");
    cout << "config file name: " << name << endl << endl;
  }
  catch(const SettingNotFoundException &nfex)
  {
    cerr << "No 'name' setting in configuration file." << endl;
  }

  const Setting& root = cfg.getRoot();

  // Output a list of all vdWFluids in the inventory.
  try
  {
    libconfig::Setting &InputImages = root["InputImages"];
    libconfig::Setting &defaultImages= InputImages["default"];
    libconfig::Setting &KCDIImages= InputImages["KCDI"];

    defaultImages.lookupValue("Intensity",common.Intensity);
    defaultImages.lookupValue("Phase",common.Phase);
    defaultImages.lookupValue("restart",common.restart);
    defaultImages.lookupValue("Pattern",common.Pattern);
    KCDIImages.lookupValue("Intensity",KCDI.Intensity);
    KCDIImages.lookupValue("Phase",KCDI.Phase);
    KCDIImages.lookupValue("restart",KCDI.restart);
    KCDIImages.lookupValue("Pattern",KCDI.Pattern);
  }
  catch(const SettingNotFoundException &nfex)
  {
    cerr << "No Image file setting in configuration file." << endl;
  }

  // Output a list of all vdWFluids in the inventory.
  try
  {
    libconfig::Setting &Job = root["Job"];
    Job.lookupValue("oversampling",oversampling);
    Job.lookupValue("saveIter",saveIter);
    Job.lookupValue("exposure",exposure);
    Job.lookupValue("shrinkThreshold",shrinkThreshold);
    Job.lookupValue("exposureKCDI",exposureKCDI);
    Job.lookupValue("lambda",lambda);
    Job.lookupValue("d",d);
    Job.lookupValue("pixelsize",pixelsize);
    Job.lookupValue("beamspotsize",beamspotsize);
    Job.lookupValue("beamStopSize",beamStopSize);
    Job.lookupValue("runSim",runSim);
    Job.lookupValue("simCCDbit",simCCDbit);
    Job.lookupValue("isFresnel",isFresnel);
    Job.lookupValue("doIteration",doIteration);
    Job.lookupValue("phaseModulation",phaseModulation);
    Job.lookupValue("useGaussionLumination",useGaussionLumination);
    Job.lookupValue("useGaussionHERALDO",useGaussionHERALDO);
    Job.lookupValue("doCentral",doCentral);
    Job.lookupValue("useRectHERALDO",useRectHERALDO);
    Job.lookupValue("doKCDI",doKCDI);
    Job.lookupValue("useDM",useDM);
    Job.lookupValue("useBS",useBS);
    Job.lookupValue("useShrinkMap",useShrinkMap);
    Job.lookupValue("reconAC",reconAC);
    Job.lookupValue("restart",restart);
    Job.lookupValue("nIterKCDI",nIterKCDI);
    Job.lookupValue("noiseLevel",noiseLevel);
    Job.lookupValue("nIter",nIter);
    Job.lookupValue("algorithm",algorithm);
  }
  catch(const SettingNotFoundException &nfex)
  {
    cerr << "No Image file setting in configuration file." << endl;
  }
}
void readConfig::print(){
  std::cout<<"common Intensity="<<common.Intensity<<std::endl;
  std::cout<<"common Phase="<<common.Phase<<std::endl;
  std::cout<<"common restart="<<common.restart<<std::endl;
  std::cout<<"common Pattern="<<common.Pattern<<std::endl;
  std::cout<<"KCDI Intensity="<<KCDI.Intensity<<std::endl;
  std::cout<<"KCDI Phase="<<KCDI.Phase<<std::endl;
  std::cout<<"KCDI restart="<<KCDI.restart<<std::endl;
  std::cout<<"KCDI Pattern="<<KCDI.Pattern<<std::endl;
  std::cout<<"oversampling="<<oversampling<<std::endl;
  std::cout<<"lambda="<<lambda<<std::endl;
  std::cout<<"d="<<d<<std::endl;
  std::cout<<"pixelsize="<<pixelsize<<std::endl;
  std::cout<<"beamspotsize="<<beamspotsize<<std::endl;
  std::cout<<"runSim="<<runSim<<std::endl;
  std::cout<<"simCCDbit="<<simCCDbit<<std::endl;
  std::cout<<"isFresnel="<<isFresnel<<std::endl;
  std::cout<<"doIteration="<<doIteration<<std::endl;
  std::cout<<"useGaussionLumination="<<useGaussionLumination<<std::endl;
  std::cout<<"useGaussionHERALDO="<<useGaussionHERALDO<<std::endl;
  std::cout<<"doCentral="<<doCentral<<std::endl;
  std::cout<<"useRectHERALDO="<<useRectHERALDO<<std::endl;
  std::cout<<"doKCDI="<<doKCDI<<std::endl;
  std::cout<<"useDM="<<useDM<<std::endl;
  std::cout<<"useBS="<<useBS<<std::endl;
  std::cout<<"useShrinkMap="<<useShrinkMap<<std::endl;
  std::cout<<"reconAC="<<reconAC<<std::endl;
  std::cout<<"restart="<<restart<<std::endl;
}

void Stringsplit(const string& str, const string& split, vector<string>& res)
{
	//std::regex ws_re("\\s+"); // 正则表达式,匹配空格 
	std::regex reg(split);		// 匹配split
	std::sregex_token_iterator pos(str.begin(), str.end(), reg, -1);
	decltype(pos) end;              // 自动推导类型 
	for (; pos != end; ++pos)
	{
		res.push_back(pos->str());
	}
}

AlgoParser::AlgoParser(std::string formula){
  printf("parsing formula: %s\n",formula.c_str());
  remove(formula.begin(),formula.end(),' ');
  auto position = formula.find("(");
  while(position!= std::string::npos){
    auto positione = formula.find(")");
    auto currentPosition = position;
    currentPosition = formula.find("(",position+1,positione-currentPosition+1);
    while(currentPosition!=std::string::npos){
      positione = formula.find(")",positione+1);
      currentPosition = formula.find("(",currentPosition+1,positione-currentPosition+1);
      std::cout<<position<<","<<currentPosition<<","<<positione<<std::endl;
    }
    subParsers.push_back(new AlgoParser(formula.substr(position+1, positione-position-1)));
    formula.replace(position, positione-position+1, "subParser");
    std::cout<<formula<<std::endl;
    position = formula.find("(");
  }
  std::vector<std::string> strs;
  Stringsplit(formula, "\\+", strs);
  printf("bracket removed: %s\n",formula.c_str());
  int iParser = 0;
  for(auto mult : strs){
    auto starpos = mult.find('*');
    int num = atoi(mult.substr(0, starpos).c_str());
    std::string str = mult.substr(starpos+1,str.size()+1);
    count.push_back(num);
    if(str=="RAAR") algoList.push_back(RAAR);
    else if(str=="HIO") algoList.push_back(HIO);
    else if(str=="ER") algoList.push_back(ER);
    else if(str=="subParser") algoList.push_back(nAlgo+iParser++);
    else{
      printf("Algorithm %s not found\n", str.c_str());
      exit(0);
    }
  }
  restart();
  for(int i = 0; i<count.size(); i++){
    printf("%d*%d,", count[i], algoList[i]);
  }
  printf("\n");
}
void AlgoParser::restart(){
  currentAlgo = 0;
  currentCount = count[0];
  for(auto sub : subParsers){
    sub->restart();
  }
}
int AlgoParser::next(){
  if(currentCount==0){
    if(currentAlgo == algoList.size()-1) return -1; // end of the algorithm
    currentCount = count[++currentAlgo];
  }
  if(algoList[currentAlgo]>=nAlgo) {
    int retVal = subParsers[algoList[currentAlgo]-nAlgo]->next();
    if(retVal==-1) { 
      currentCount--;
      subParsers[algoList[currentAlgo]-nAlgo]->restart();
      return subParsers[algoList[currentAlgo]-nAlgo]->next();
    }
    return retVal;
  } else {
    currentCount--;
    return algoList[currentAlgo];
  }
}
// eof
