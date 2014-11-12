// Distributed under GNU General Public License (see license.txt for details).
//
//  Copyright (c) 2007 Shai Shalev-Shwartz.
//  All Rights Reserved.
//
// Main for the PEGASOS algorithm.
//
//
// 
// Pegasos: Fly high in the primal 
// main file
//
// 

//*****************************************************************************
// Included Files
//*****************************************************************************
#include "cmd_line.h"
#include "simple_sparse_vec_hash.h"
#include "pegasos_optimize.h"



int main(int argc, char** argv) {


    // -------------------------------------------------------------
    // ---------------------- Parse Command Line -------------------
    // -------------------------------------------------------------

    std::string data_filename;
    std::string test_filename;
    std::string model_filename;
    std::string experiments_file = "noExperimentsFile";
    double lambda = 1.0;
    int max_iter = 10;
    int exam_per_iter = 1;
    //uint num_iter_to_avg = 100;

    // parse command line
    learning::cmd_line cmdline;
    cmdline.info("Pegasos algorithm");
    cmdline.add_master_option("<data-file>", &data_filename);
    cmdline.add("-lambda", "regularization parameter (default = 0.01)", &lambda, 0.001);
    cmdline.add("-iter", "number of iterations (default = 10/lambda)", &max_iter, int(100/lambda));
    cmdline.add("-k", "size of block for stochastic gradient (default = 1)", &exam_per_iter, 1);
    cmdline.add("-modelFile","name of model file (default = noModelFile)", &model_filename,"noModelFile");
    cmdline.add("-testFile","name of test data file (default = noTestFile)", &test_filename,"noTestFile");
    //cmdline.add("-uniform","test under uniform condition (default = 0)", &uni, 0);
    //   cmdline.add("-experiments","name of experiments spec. file", 
    // 	      &experiments_file,"noExperimentsFile");

    int rc = cmdline.parse(argc, argv);
    if (rc < 2) {
        cmdline.print_help();
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------
    // ---------------------- read the data ------------------------
    // -------------------------------------------------------------
    uint dimension = 0;
    std::vector<simple_sparse_vector> Dataset;
    std::vector<int> Labels;
    long readingTime;
    ReadData(data_filename,Dataset,Labels,dimension,readingTime);

    uint testDimension = 0;
    std::vector<simple_sparse_vector> testDataset;
    std::vector<int> testLabels;
    long testReadingTime;
    if (test_filename != "noTestFile") {
        ReadData(test_filename,testDataset,testLabels,testDimension,testReadingTime);
    } else {
        testReadingTime = 0;
    }
    std::cerr << readingTime+testReadingTime << " = Time for reading the data" <<  std::endl;

    // choose a random seed
    srand(time(NULL));


    // -------------------------------------------------------------
    // ---------------------- Experiments mode ---------------------
    // -------------------------------------------------------------
    /*
       if (experiments_file != "noExperimentsFile") {
       run_experiments(experiments_file,Dataset,Labels,dimension,
       testDataset,testLabels);
       return(EXIT_SUCCESS); 
       }
       */

    // -------------------------------------------------------------
    // ---------------------- Main Learning function ---------------
    // -------------------------------------------------------------
    long trainTime,calc_obj_time;
    double obj_value,norm_value,loss_value,zero_one_error,test_loss,test_error;
    const int ROUND = 100;
    double total_obj = 0, total_terr = 0;

    //Calculate all the p_i
    std::vector<double> p;
    p.clear();
    double sumup = 0;
    double average = 0;
    double variance = 0;
    uint num_examples = Labels.size();

    //for non-uniform sampling
    for (uint i = 0; i < num_examples; ++i) {
        p.push_back(sqrt(Dataset[i].snorm()));
        sumup += p[i];
    }
    average = sumup / num_examples;
    for(uint i = 0; i < num_examples; ++i) {
        variance += (average-p[i])*(average-p[i]);
    }
    variance = variance / num_examples;
    std::cout << "Norm variance = " << variance << std::endl;

    for (uint i = 0; i < num_examples; ++i) {
        p[i] /= sumup;
    }


    for(int rounds = 1; rounds <= ROUND; rounds++) {
        LearnReturnLast(Dataset,Labels,dimension,testDataset,testLabels,
                lambda,max_iter,exam_per_iter,
                model_filename, p,
                trainTime,calc_obj_time,obj_value,norm_value,
                loss_value,zero_one_error,
                test_loss,test_error,
                0,0.0,0,0.0);
        total_obj = total_obj + obj_value;
        total_terr = total_terr + test_error;

    }

    //std::cout << trainTime / ROUND << " = Time for training\n" 
    //      << calc_obj_time / ROUND  << " = Time for calculate objective\n" 
    //      << norm_value / ROUND  << " = Norm of solution\n" 
    //      << loss_value / ROUND << " = avg Loss of solution\n"  
    //      << zero_one_error / ROUND  << " = avg zero-one error of solution\n" 
    std::cout << total_obj / ROUND << " = primal objective of solution\n" 
        //<< test_loss / ROUND << " = avg Loss over test\n"  
        << total_terr / ROUND  << " = avg zero-one error over test\n" 	    
        <<  std::endl;


    p.clear();
    //for unifrom sampling
    for (uint i = 0; i < num_examples; ++i) {
        p.push_back(1.0/num_examples);
    }

    total_obj = 0;
    total_terr = 0;
    for(int rounds = 1; rounds <= ROUND; rounds++) {
        LearnReturnLast(Dataset,Labels,dimension,testDataset,testLabels,
                lambda,max_iter,exam_per_iter,
                model_filename, p,
                trainTime,calc_obj_time,obj_value,norm_value,
                loss_value,zero_one_error,
                test_loss,test_error,
                0,0.0,0,0.0);
        total_obj = total_obj + obj_value;
        total_terr = total_terr + test_error;
    }


    //std::cout << trainTime / ROUND << " = Time for training\n" 
    //      << calc_obj_time / ROUND  << " = Time for calculate objective\n" 
    //      << norm_value / ROUND  << " = Norm of solution\n" 
    //      << loss_value / ROUND << " = avg Loss of solution\n"  
    //      << zero_one_error / ROUND  << " = avg zero-one error of solution\n" 
    std::cout << total_obj / ROUND << " = primal objective of solution\n" 
        //<< test_loss / ROUND << " = avg Loss over test\n"  
        << total_terr / ROUND  << " = avg zero-one error over test\n" 	    
        <<  std::endl;

    return(EXIT_SUCCESS);
}

