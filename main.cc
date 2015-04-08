//*****************************************************************************
// Copyright [2015] <Raynald Chung>
// Included Files
//*****************************************************************************
#include "include/cmd_line.h"
#include "include/simple_sparse_vec_hash.h"
#include "include/optimize.h"

int main(int argc, char** argv) {
    // -------------------------------------------------------------
    // ---------------------- Parse Command Line -------------------
    // -------------------------------------------------------------

    std::string data_filename;
    std::string test_filename;
    std::string model_filename;
    std::string experiments_file = "noExperimentsFile";
    double lambda = 0.0001;
    uint epoch = 30;
    uint num_rounds = 10;

    // parse command line
    learning::cmd_line cmdline;
    cmdline.info("Non-uniform SGD and SDCA algorithm");
    cmdline.add_master_option("<data-file>", &data_filename);
    cmdline.add("-epoch", "number of epoches (default = 30)", &epoch, 30);
    cmdline.add("-round", "number of round (default = 10)", &num_rounds, 10);
    cmdline.add("-lambda", "regularization parameter (default = 0.0001)", &lambda, 0.0001);
    cmdline.add("-testFile", "name of test data file (default = noTestFile)", &test_filename,"noTestFile");

    int rc = cmdline.parse(argc, argv);
    if (rc < 2) {
        cmdline.print_help();
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------
    // ---------------------- read the data ------------------------ // ------------------------------------------------------------- 
    uint dimension = 0;
    std::vector<simple_sparse_vector> Dataset;
    std::vector<int> Labels;
    double readingTime;
    Model mod;

    mod.ReadData(data_filename, Dataset, Labels, dimension, readingTime);

    uint testDimension = 0;
    std::vector<simple_sparse_vector> testDataset;
    std::vector<int> testLabels;
    double testReadingTime;
    if (test_filename != "noTestFile") {
        mod.ReadData(test_filename, testDataset, testLabels, testDimension, testReadingTime);
    } else {
        testReadingTime = 0;
    }
    std::cerr << readingTime + testReadingTime << " = Time for reading the data" <<  std::endl;

    // choose a random seed
    srand(20141225);

    // -------------------------------------------------------------
    // ---------------------- Main Learning function ---------------
    // -------------------------------------------------------------
    // calculate all the p_i
    std::vector<double> p;
    p.clear();
    double sumup = 0;
    double average = 0;
    double variance = 0;
    uint num_examples = Labels.size();

    // for adaptive sampling
    p.push_back(1);
    for (uint i = 1; i <= num_examples; ++i) {
        p.push_back(sqrt(Dataset[i].snorm()) + sqrt(lambda));
        sumup += p[i];
    }
    average = sumup / num_examples;
    for(uint i = 0; i < num_examples; ++i) {
        variance += (average - p[i]) * (average - p[i]);
    }
    variance = variance / num_examples;
    std::cout << "Norm average = " << average << std::endl;
    std::cout << "Norm variance = " << variance << std::endl;
    std::cout << "Num examples = " << num_examples << std::endl;
    std::cout << "Num rounds = " << num_rounds << std::endl;
    std::cout << "Num epochs = " << epoch << std::endl;
    std::cout << "Lambda = " << lambda << std::endl;

    for (uint i = 1; i <= num_examples; ++i) {
        p[i] /= sumup;
    }


    //SDCA parameter: is_daptive, ada_rule_type, online
    /*
    std::cout << "AdaSGD:\n";
    mod.SGDLearn(Dataset, Labels, dimension, testDataset, testLabels,
            lambda, p, Adaptive, 0, num_rounds, epoch, 1);

    std::cout << "AdaSGD2:\n";
    mod.SGDLearn(Dataset, Labels, dimension, testDataset, testLabels,
            lambda, p, Adaptive2, 0, num_rounds, epoch, 1);

    std::cout << "AdaSDCA2:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Adaptive2, num_rounds, epoch, 1);

            */
    /*
    std::cout << "AdaSDCA3:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Adaptive, num_rounds, epoch, 1);
            */


    /*
    std::cout << "AdaSDCA+:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, AdaSDCAp, num_rounds, epoch, 1);

    std::cout << "OASGD:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Online, 0, num_rounds, epoch, 1);


    std::cout << "OASDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Online2, num_rounds, epoch, 1);

    std::cout << "NonUnifSGD:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, 0, num_rounds, epoch, 1);


    std::cout << "NonUnifSDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, num_rounds, epoch, 1);
            */
 
    std::cout << "AdaGrad:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, AdaGrad, 0, num_rounds, epoch, 1);

    std::cout << "AdaSVRG:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, VarianceReduction, 0, num_rounds, epoch, 1);

    /*
    p.clear();
    p.push_back(1);
    sumup = 0;
    for (uint i = 1; i <= num_examples; ++i) {
        p.push_back(sqrt(Dataset[i].snorm()));
        sumup += p[i];
    }
    for (uint i = 1; i <= num_examples; ++i) {
        p[i] /= sumup;
    }

    std::cout << "Non-uniform SGD 2:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, 0, num_rounds, epoch);

    std::cout << "Non-uniform SDCA 2:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, num_rounds, epoch);
 
    p.clear();
    p.push_back(1);
    sumup = 0;
    for (uint i = 1; i <= num_examples; ++i) {
        p.push_back(1.0 / sqrt(Dataset[i].snorm()));
        sumup += p[i];
    }
    for (uint i = 1; i <= num_examples; ++i) {
        p[i] /= sumup;
    }

    std::cout << "Non-uniform SGD 3:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, 0, num_rounds, epoch);

    std::cout << "Non-uniform SDCA 3:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, num_rounds, epoch);
 
    */
 
    //SGD parameter: Adaptive, VarianceReduction, Online, AdaGrad
    //eta_rule_type
    /*
    p.clear();
    p.push_back(1);
    for (uint i = 0; i < num_examples; ++i) {
        p.push_back(1.0 / num_examples);
    }
     std::cout << "UnifSGD:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, 0, num_rounds, epoch, 1);


    std::cout << "UnifSDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, num_rounds, epoch, 1);
            */
 
    //SDCA parameter: is_daptive, ada_rule_type, online
    /*
    std::cout << "Adaptive SGD:\n";
    mod.SGDLearn(Dataset, Labels, dimension, testDataset, testLabels,
            lambda, p, Adaptive, 0, num_rounds, epoch);
    */

    /*
    std::cout << "Plain SGD:\n";
    mod.SGDLearn(Dataset, Labels, dimension, testDataset, testLabels,
            lambda, p, Plain, 0, num_rounds, epoch, 1);

    std::cout << "Plain SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, num_rounds, epoch, 1);
    */
 
    /*
     std::cout << "AdaGrad SGD:\n";
    mod.SGDLearn(Dataset, Labels, dimension, testDataset, testLabels,
            lambda, p, AdaGrad, 0, num_rounds, epoch);
            */
    

    /*
    std::cout << "Adaptive SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Adaptive, num_rounds, epoch);
    */

    /*
    std::cout << "Adaptive SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, 1, 1, 0, num_rounds, epoch);
    */

    /*
    std::cout << "Adaptive SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, 1, 1, 1, num_rounds, epoch);
    */


    /*
    std::cout << "Adaptive 1/t SGD:\n";
    mod.SGDLearn(Dataset, Labels, dimension, testDataset, testLabels,
            lambda, p, 1, 0, 1, num_rounds, epoch);
    */
 
    //for non-uniform sampling
    /*
    std::cout << "Non-uniform SGD:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, 0, num_rounds, epoch);
    */

    /*
    std::cout << "Non-uniform SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, 0, 0, 0, num_rounds, epoch);
    */


    //for unifrom sampling
    /*
    p.clear();
    p.push_back(1);
    for (uint i = 0; i < num_examples; ++i) {
        p.push_back(1.0 / num_examples);
    }
    */ 
    /*
    std::cout << "Uniform SGD:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Online, 0, num_rounds, epoch);

    */
    /*
    std::cout << "Uniform SGD:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, 0, num_rounds, epoch);
    */


    /*
    std::cout << "Uniform SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Online, num_rounds, epoch);

     std::cout << "Uniform SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Online2, num_rounds, epoch);

      std::cout << "Uniform SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Plain, num_rounds, epoch);
            */

    /*
       std::cout << "Uniform SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Adaptive, num_rounds, epoch);

        std::cout << "Uniform SDCA:\n";
    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, Adaptive2, num_rounds, epoch);
    
            */
    //Variance reduction
    /*
    std::cout << "Variance SGD:\n";
    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda, p, 0, 1, 0, 0, num_rounds, epoch);
    */
 
    return(EXIT_SUCCESS);
}

