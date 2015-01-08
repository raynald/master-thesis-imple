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
     const int kMAXEPOCH = 110;
     const int kNUMEXAMPLE = 465000;

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
             // Output variables
             double& train_time, double& calc_obj_time,
             double& obj_value, double& norm_value,
             double& loss_value, double& zero_one_error,
             double& test_loss, double& test_error,
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
             // Output variables
             double& train_time, double& calc_obj_time,
             double& obj_value, double& norm_value,
             double& loss_value, double& zero_one_error,
             double& test_loss, double& test_error,
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
