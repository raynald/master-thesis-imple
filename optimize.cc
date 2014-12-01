// Distributed under GNU General Public License (see license.txt for details).
//
//  Copyright (c) 2007 Shai Shalev-Shwartz.
//  All Rights Reserved.
//=============================================================================
// File Name: pegasos_optimize.cc
// implements the main optimization function of pegasos
//=============================================================================

#include "optimize.h"

// help function for getting runtime
long get_runtime(void) {
    clock_t start;
    start = clock();
    return((long)((double)start/(double)CLOCKS_PER_SEC));
}


uint get_sample(std::vector<double> &p) {
    double x = rand() * 1.0 / RAND_MAX;
    for(uint i=0;i<p.size();i ++) {
        if(x<p[i]) return i; else x-=p[i];
    }
    return p.size()-1;
}

// ------------------------------------------------------------//
// ---------------- OPTIMIZING --------------------------------//
// ------------------------------------------------------------//

void LearnReturnLast(// Input variables
        std::vector<simple_sparse_vector> Dataset,
        std::vector<int> Labels,
        uint dimension,
        std::vector<simple_sparse_vector> testDataset,
        std::vector<int> testLabels,
        double lambda,int max_iter,
        int exam_per_iter,
        std::string& model_filename,
        std::vector<double> &p,
        bool change, 
        // Output variables
        long& train_time,long& calc_obj_time,
        double& obj_value,double& norm_value,
        double& loss_value,double& zero_one_error,
        double& test_loss,double& test_error,
        // additional parameters
        int eta_rule_type , double eta_constant ,
        int projection_rule, double projection_constant) {

    uint num_examples = Labels.size();

    long startTime = get_runtime();
    long endTime;

    double chiv[num_examples];
    double sumup;
    uint chiv_size[num_examples];

    for(uint i=0;i<num_examples;i++) {
        chiv[i] = 0;//1.0/num_examples;
        chiv_size[i] = 0;
    }

    // Initialization of classification vector
    WeightVector W(dimension);

    // ---------------- Main Loop -------------------
    max_iter = 10 * num_examples;
    std::cout << max_iter<<std::endl;
    for (int i = 0; i < max_iter; ++i) {

        if (change && (i+1) % num_examples == 0) {
            //begin of each epoch, update p
            sumup = 0;
            for(uint j=0; j<num_examples; j++) {
                p[j] = chiv[j];
                sumup += chiv[j];
                chiv_size[j] = 0;
            }
            std::cout << "fu"<<sumup <<std::endl;
            for(uint j=0; j<num_examples; j++) {
                p[j] /= sumup;
            }
        }
        // learning rate
        double eta;
        if (eta_rule_type == 0) { // Pegasos eta rule
            eta = 1 / (lambda * (i+2)); 
        } else if (eta_rule_type == 1) { // Norma rule
            eta = eta_constant / sqrt(i+2);
            // solve numerical problems
            //if (projection_rule != 2)
            W.make_my_a_one();
        } else {
            eta = eta_constant;
        }

        // gradient indices and losses
        std::vector<uint> grad_index;
        std::vector<double> grad_weights;

        // calc sub-gradients
        //for (int j=0; j < exam_per_iter; ++j) {

        // choose random example
        uint r = get_sample(p);
        //((int)rand()) % num_examples;

        // calculate prediction
        double prediction = W*Dataset[r];

        // calculate loss
        double cur_loss = 1 - Labels[r]*prediction;
        if (cur_loss < 0.0) cur_loss = 0.0;

        // and add to the gradient
        if (cur_loss > 0.0) {
            grad_index.push_back(r);
            grad_weights.push_back(eta*Labels[r]/exam_per_iter/num_examples/p[r]);
        }
        WeightVector tmp1(dimension);
        tmp1 = W;
        simple_sparse_vector tmp2 = Dataset[r];
        tmp1.scale(lambda);
        //tmp2.scale(Labels[r]);
        tmp1.add(tmp2, Labels[r]);
        tmp1.scale(eta/num_examples/p[r]);
        double temp = sqrt(tmp1.snorm());
        chiv[r] = (chiv[r]*chiv_size[r]+temp)/(chiv_size[r]+1);
        chiv_size[r]++;
        //}

        // scale w 
        W.scale(1.0 - eta*lambda/num_examples/p[r]);

        // and add sub-gradients
        for (uint j=0; j<grad_index.size(); ++j) {
            W.add(Dataset[grad_index[j]],grad_weights[j]);
        }
    }


    // update timeline
    endTime = get_runtime();
    train_time = endTime - startTime;
    startTime = get_runtime();

    // Calculate objective value
    norm_value = W.snorm();
    obj_value = norm_value * lambda / 2.0;
    loss_value = 0.0;
    zero_one_error = 0.0;
    for (uint i=0; i < Dataset.size(); ++i) {
        double cur_loss = 1 - Labels[i]*(W * Dataset[i]); 
        if (cur_loss < 0.0) cur_loss = 0.0;
        loss_value += cur_loss/num_examples;
        obj_value += cur_loss/num_examples;
        if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
    }

    endTime = get_runtime();
    calc_obj_time = endTime - startTime;

    // Calculate test_loss and test_error
    test_loss = 0.0;
    test_error = 0.0;
    for (uint i=0; i < testDataset.size(); ++i) {
        double cur_loss = 1 - testLabels[i]*(W * testDataset[i]); 
        if (cur_loss < 0.0) cur_loss = 0.0;
        test_loss += cur_loss;
        if (cur_loss >= 1.0) test_error += 1.0;
    }
    if (testDataset.size() != 0) {
        test_loss /= testDataset.size();
        test_error /= testDataset.size();
    }



    // finally, print the model to the model_file
    if (model_filename != "noModelFile") {
        std::ofstream model_file(model_filename.c_str());
        if (!model_file.good()) {
            std::cerr << "error w/ " << model_filename << std::endl;
            exit(EXIT_FAILURE);
        }
        W.print(model_file);
        model_file.close();
    }

}




// ------------------------------------------------------------//
// ---------------- READING DATA ------------------------------//
// ------------------------------------------------------------//
void ReadData(// input
        std::string& data_filename,
        // output
        std::vector<simple_sparse_vector> & Dataset,
        std::vector<int> & Labels,
        uint& dimension,
        long& readingTime) {

    dimension = 0;

    // Start a timer
    long startTime = get_runtime();

    // OPEN DATA FILE
    // =========================
    std::ifstream data_file(data_filename.c_str());
    if (!data_file.good()) {
        std::cerr << "error w/ " << data_filename << std::endl;
        exit(EXIT_FAILURE);
    }


    // Read SVM-Light data file
    // ========================
    int num_examples = 0;
    std::string buf;
    while (getline(data_file,buf)) {
        // ignore lines which begin with #
        if (buf[0] == '#') continue;
        // Erase what comes after #
        size_t pos = buf.find('#');
        if (pos < buf.size()) {
            buf.erase(pos);
        }
        // replace ':' with white space
        int n=0;
        for (size_t pos=0; pos < buf.size(); ++pos)
            if (buf[pos] == ':') {
                n++; buf[pos] = ' ';
            }
        // read from the string
        std::istringstream is(buf);
        int label = 0;
        is >> label;
        /*
        if (label != 1 && label != -1) {
            std::cerr << "Error reading SVM-light format. Abort." << std::endl;
            exit(EXIT_FAILURE);
        }
        */
        if(label==0) label=-1;
        Labels.push_back(label);
        simple_sparse_vector instance(is,n);
        Dataset.push_back(instance);
        num_examples++;
        uint cur_max_ind = instance.max_index() + 1;
        if (cur_max_ind > dimension) dimension = cur_max_ind;
    }

    data_file.close();


#ifdef nodef
    std::cerr << "num_examples = " << num_examples 
        << " dimension = " << dimension
        << " Dataset.size = " << Dataset.size() 
        << " Labels.size = " << Labels.size() << std::endl;
#endif


    // update timeline
    readingTime = get_runtime() - startTime;

}



// -------------------------------------------------------------//
// ---------------------- Experiments mode ---------------------//
// -------------------------------------------------------------//

class ExperimentStruct {
    public:
        ExperimentStruct() { }
        void Load(std::istringstream& is) {
            is >> lambda >> max_iter >> exam_per_iter >> num_iter_to_avg
                >> eta_rule >> eta_constant >> projection_rule 
                >> projection_constant;
        }
        void Print() {
            std::cout << lambda << "\t\t" << max_iter << "\t\t" << exam_per_iter << "\t\t" 
                << num_iter_to_avg << "\t\t" << eta_rule << "\t\t" << eta_constant 
                << "\t\t" << projection_rule << "\t\t" << projection_constant << "\t\t";
        }
        void PrintHead() {
            std::cerr << "lambda\t\tT\t\tk\t\tnumValid\t\te_r\t\te_c\t\tp_r\t\tp_c\t\t";
        }
        double lambda, eta_constant, projection_constant;
        uint max_iter,exam_per_iter,num_iter_to_avg;
        int eta_rule,projection_rule;
};

class ResultStruct {
    public:
        ResultStruct() : trainTime(0), calc_obj_time(0),
        norm_value(0.0),loss_value(0.0),
        zero_one_error(0.0),obj_value(0.0),
        test_loss(0.0), test_error(0.0) { }
        void Print() {
            std::cout << trainTime << "\t\t" 
                << calc_obj_time << "\t\t" 
                << norm_value  << "\t\t" 
                << loss_value << "\t\t"  
                << zero_one_error  << "\t\t" 
                << obj_value << "\t\t"
                << test_loss << "\t\t"
                << test_error << "\t\t";

        }
        void PrintHead() {
            std::cerr << "tTime\t\tcoTime\t\t||w||\t\tL\t\tL0-1\t\tobj_value\t\ttest_L\t\ttestL0-1\t\t";
        }

        long trainTime, calc_obj_time;
        double norm_value,loss_value,zero_one_error,obj_value,test_loss,test_error;
};

/*
void run_experiments(std::string& experiments_filename,
        std::vector<simple_sparse_vector>& Dataset,
        std::vector<int>& Labels,
        uint dimension,
        std::vector<simple_sparse_vector>& testDataset,
        std::vector<int>& testLabels) {

    // open the experiments file
    std::ifstream exp_file(experiments_filename.c_str());
    if (!exp_file.good()) {
        std::cerr << "error w/ " << experiments_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // read the entire experiment specification


    uint num_experiments = 0;
    std::vector<ExperimentStruct> v;
    std::string buf;
    while (getline(exp_file,buf)) {
        // read from the string
        std::istringstream is(buf);
        ExperimentStruct tmp; tmp.Load(is);
        v.push_back(tmp);
        num_experiments++;
    }
    exp_file.close();

    // run all the experiments
    std::vector<ResultStruct> res(num_experiments);
    std::string lala = "noModelFile";
    for (uint i=0; i<num_experiments; ++i) {
        LearnReturnLast(Dataset,Labels,dimension,testDataset,testLabels,
                v[i].lambda,v[i].max_iter,v[i].exam_per_iter,
                //v[i].num_iter_to_avg,
                lala,
                res[i].trainTime,res[i].calc_obj_time,res[i].obj_value,
                res[i].norm_value,
                res[i].loss_value,res[i].zero_one_error,
                res[i].test_loss,res[i].test_error,
                v[i].eta_rule,v[i].eta_constant,
                v[i].projection_rule,v[i].projection_constant);
    }

    // print results
    v[0].PrintHead(); res[0].PrintHead(); 
    std::cout << std::endl;
    for (uint i=0; i<num_experiments; ++i) {
        v[i].Print();
        res[i].Print();
        std::cout << std::endl;
    }

    std::cerr << std::endl;

}
*/
