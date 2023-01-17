#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "mnistData.h"
#define mn torch::data::datasets::MNIST
#define dat ((mn*)dataset)
mnistData::mnistData(const char* dir, int &row, int &col){
  dataset = new mn(dir);
  auto theData = dat->get(0).data;
  auto foo_a = theData.accessor<float,3>();
  row = foo_a[0].size(0);
  col = foo_a[0][0].size(0);
  idat = 0;
}
Real* mnistData::read(){
  if(idat == dat->size().value()) return 0;
  auto theData = dat->get(idat).data;
  auto foo_a = theData.accessor<float,3>();
  idat++;
  return &(foo_a[0][0][0]);
}
mnistData::~mnistData(){};
