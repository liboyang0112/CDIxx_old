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
  cout << "config file: " << configfile << endl;
  if(ret==EXIT_FAILURE) exit(ret);

  const Setting& root = cfg.getRoot();

  // Output a list of all vdWFluids in the inventory.
  try
  {
    libconfig::Setting &InputImages = root["InputImages"];
    libconfig::Setting &defaultImages= InputImages["default"];
    libconfig::Setting &pupilImages= InputImages["pupil"];

    defaultImages.lookupValue("Intensity",common.Intensity);
    defaultImages.lookupValue("Phase",common.Phase);
    defaultImages.lookupValue("restart",common.restart);
    defaultImages.lookupValue("Pattern",common.Pattern);
    pupilImages.lookupValue("Intensity",pupil.Intensity);
    pupilImages.lookupValue("Phase",pupil.Phase);
    pupilImages.lookupValue("restart",pupil.restart);
    pupilImages.lookupValue("Pattern",pupil.Pattern);
  }
  catch(const SettingNotFoundException &nfex)
  {
    cerr << "No Image file setting in configuration file." << endl;
  }

  // Output a list of all vdWFluids in the inventory.
  try
  {
#define getVal(x,y) Job.lookupValue(#x,x);
    libconfig::Setting &Job = root["Job"];
    BOOLVAR(getVal)
    INTVAR(getVal)
    REALVAR(getVal)
    STRVAR(getVal)
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
  std::cout<<"pupil Intensity="<<pupil.Intensity<<std::endl;
  std::cout<<"pupil Phase="<<pupil.Phase<<std::endl;
  std::cout<<"pupil restart="<<pupil.restart<<std::endl;
  std::cout<<"pupil Pattern="<<pupil.Pattern<<std::endl;

#define PRINTBOOL(x,y) std::cout<<"bool: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
#define PRINTINT(x,y) std::cout<<"int: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
#define PRINTREAL(x,y) std::cout<<"flost: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
#define PRINTSTR(x,y) std::cout<<"string: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
    BOOLVAR(PRINTBOOL)
    INTVAR(PRINTINT)
    REALVAR(PRINTREAL)
    STRVAR(PRINTSTR)
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
