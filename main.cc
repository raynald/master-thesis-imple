//*****************************************************************************
// Included Files
//*****************************************************************************
#include "cmd_line.h"
#include "simple_sparse_vec_hash.h"
#include "optimize.h"



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

    // parse command line
    learning::cmd_line cmdline;
    cmdline.info("Non-uniform SGD and SDCA algorithm");
    cmdline.add_master_option("<data-file>", &data_filename);
    cmdline.add("-lambda", "regularization parameter (default = 0.01)", &lambda, 0.001);
    cmdline.add("-iter", "number of iterations (default = 10/lambda)", &max_iter, int(100/lambda));
    cmdline.add("-testFile","name of test data file (default = noTestFile)", &test_filename,"noTestFile");

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
    long readingTime;
    Model mod;

    mod.ReadData(data_filename,Dataset,Labels,dimension,readingTime);

    uint testDimension = 0;
    std::vector<simple_sparse_vector> testDataset;
    std::vector<int> testLabels;
    long testReadingTime;
    if (test_filename != "noTestFile") {
        mod.ReadData(test_filename,testDataset,testLabels,testDimension,testReadingTime);
    } else {
        testReadingTime = 0;
    }
    std::cerr << readingTime+testReadingTime << " = Time for reading the data" <<  std::endl;

    // choose a random seed
    srand(20141010);

    // -------------------------------------------------------------
    // ---------------------- Main Learning function ---------------
    // -------------------------------------------------------------
    long trainTime,calc_obj_time;
    double obj_value,norm_value,loss_value,zero_one_error,test_loss,test_error;

    //Calculate all the p_i
    std::vector<double> p;
    p.clear();
    double sumup = 0;
    double average = 0;
    double variance = 0;
    uint num_examples = Labels.size();

    //for non-uniform sampling
    for (uint i = 0; i < num_examples; ++i) {
        p.push_back(sqrt(Dataset[i].snorm())+sqrt(lambda));
        sumup += p[i];
    }
    average = sumup / num_examples;
    for(uint i = 0; i < num_examples; ++i) {
        variance += (average-p[i])*(average-p[i]);
    }
    variance = variance / num_examples;
    std::cout << "Norm average = " << average << std::endl;
    std::cout << "Norm variance = " << variance << std::endl;

    for (uint i = 0; i < num_examples; ++i) {
        p[i] /= sumup;
        //p[i] = 1.0/ num_examples;
    }


    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 1, 
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            0);

    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 1, 
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            1);

    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 1, 
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            0);

    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 1, 
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            1);


    p.clear();
    //for unifrom sampling
    for (uint i = 0; i < num_examples; ++i) {
        p.push_back(1.0/num_examples);
    }

    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 0,
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            0);

    mod.SGDLearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 0,
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            1);

    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 0,
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            0);

    mod.SDCALearn(Dataset,Labels,dimension,testDataset,testLabels,
            lambda,max_iter, p, 0,
            trainTime,calc_obj_time,obj_value,norm_value,
            loss_value,zero_one_error,
            test_loss,test_error,
            1);


    return(EXIT_SUCCESS);
}

