//=============================================================================
// File Name: optimize.h
// header for the main optimization function of pegasos
//=============================================================================

#ifndef _OPTIMIZE_H
#define _OPTIMIZE_H

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
#include "simple_sparse_vec_hash.h"
#include "WeightVector.h"


// function from Joachims that measures CPU time
long get_runtime(void);

// main optimization function

class Model {
    public:
        // main optimization function for SGD
        void SGDLearn(// Input variables
		      std::vector<simple_sparse_vector> Dataset,
		      std::vector<int> Labels,
		      uint dimension,
		      std::vector<simple_sparse_vector> testDataset,
		      std::vector<int> testLabels,
		      double lambda,
              std::vector<double> p, 
              bool change, bool VR,
		      // Output variables
		      long& train_time,long& calc_obj_time,
		      double& obj_value,double& norm_value,
		      double& loss_value,double& zero_one_error,
		      double& test_loss,double& test_error,
		      // additional parameters
		      int eta_rule_type, const uint num_round, const uint num_epoch);

        // main optimization function for SDCA
        void SDCALearn(// Input variables
		      std::vector<simple_sparse_vector> Dataset,
		      std::vector<int> Labels,
		      uint dimension,
		      std::vector<simple_sparse_vector> testDataset,
		      std::vector<int> testLabels,
		      double lambda,
              std::vector<double> p,
              bool change,
		      // Output variables
		      long& train_time,long& calc_obj_time,
		      double& obj_value,double& norm_value,
		      double& loss_value,double& zero_one_error,
		      double& test_loss,double& test_error,const uint num_round, const uint num_epoch);

        void localSDCA(// Input variables
		      std::vector<simple_sparse_vector> Dataset,
		      std::vector<int> Labels,
		      uint dimension,
		      std::vector<simple_sparse_vector> testDataset,
		      std::vector<int> testLabels,
		      double lambda,
              std::vector<double> p,
              bool change,
		      // Output variables
		      long& train_time,long& calc_obj_time,
		      double& obj_value,double& norm_value,
		      double& loss_value,double& zero_one_error,
		      double& test_loss,double& test_error,const uint num_round, const uint num_epoch);

 
        // function for reading the data
        void ReadData(// input
                std::string& data_filename,
                // output
                std::vector<simple_sparse_vector> & Dataset,
                std::vector<int> & Labels,
                uint& dimension,
                long& readingTime);

};
#endif
