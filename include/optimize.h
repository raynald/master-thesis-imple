//=============================================================================
// File Name: optimize.h
// header for the main optimization function of pegasos
// Copyright [2015] <Raynald Chung>
//=============================================================================

#ifndef INCLUDE_OPTIMIZE_H_
#define INCLUDE_OPTIMIZE_H_

//*****************************************************************************
// Included Files
//*****************************************************************************
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <ctime>
#include <cstring>
#include <cmath>
#include <string>
#include "simple_sparse_vec_hash.h"
#include "WeightVector.h"


// function from Joachims that measures CPU time
double get_runtime(void);

// main optimization function

class Model {
 private:
     struct ResultStruct {
         double train_time;
         double calc_obj_time;
         double norm_value;
         double loss_value;
         double zero_one_error;
         double obj_value;
         double test_loss;
         double test_error;
         ResultStruct() {
             train_time = 0;
             calc_obj_time = 0;
             norm_value = 0;
             loss_value = 0;
             zero_one_error = 0;
             obj_value = 0;
             test_loss = 0;
             test_error = 0;
         }
     };

 public:
     // Main optimization function for SGD
     void SGDLearn(
             // Input variables
             std::vector<simple_sparse_vector> Dataset,
             std::vector<int> Labels,
             uint dimension,
             std::vector<simple_sparse_vector> testDataset,
             std::vector<int> testLabels,
             double lambda,
             std::vector<double> p,
             bool is_adaptive, bool use_variance_reduction,
             // Additional parameters
             int eta_rule_type, const uint &num_round, const uint &num_epoch);

     // Main optimization function for SDCA
     void SDCALearn(
             // Input variables
             std::vector<simple_sparse_vector> Dataset,
             std::vector<int> Labels,
             uint dimension,
             std::vector<simple_sparse_vector> testDataset,
             std::vector<int> testLabels,
             double lambda,
             std::vector<double> p,
             bool is_adaptive,
             // Additional parameters
             const uint &num_round, const uint &num_epoch);

     // Function for reading the data
     void ReadData(
             // Input
             std::string& data_filename,
             // Output
             std::vector<simple_sparse_vector> & Dataset,
             std::vector<int> & Labels,
             uint& dimension,
             double& readingTime);
};

#endif  // INCLUDE_OPTIMIZE_H_
