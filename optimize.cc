//=============================================================================
// File Name: optimize.cc
// implements the main optimization function
//=============================================================================

#include "optimize.h"
#include <algorithm>

using namespace std;

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

void Model::SGDLearn(// Input variables
        std::vector<simple_sparse_vector> Dataset,
        std::vector<int> Labels,
        uint dimension,
        std::vector<simple_sparse_vector> testDataset,
        std::vector<int> testLabels,
        double lambda,
        std::vector<double> &p,
        bool change, 
        // Output variables
        long& train_time,long& calc_obj_time,
        double& obj_value,double& norm_value,
        double& loss_value,double& zero_one_error,
        double& test_loss,double& test_error,
        // additional parameters
        int eta_rule_type, const uint num_round, const uint num_epoch) {

    uint num_examples = Labels.size();

    long startTime = get_runtime();
    long endTime;

    double chiv[num_examples];
    double count[num_examples];
    double obj[num_epoch];
    double test[num_epoch];
    double t;
    double cur_loss;

    WeightVector W(dimension);
    WeightVector weight_W(dimension);
    memset(obj, 0, sizeof(obj));
    memset(test, 0, sizeof(test));
    // ---------------- Main Loop -------------------
    cout << "num_examples: " << num_examples << endl;
    for(uint round = 1; round <= num_round; round++) {
        W.scale(0);
        weight_W.scale(0);
        t = 0;
        for(uint epoch = 0;epoch < num_epoch; epoch++) {
            memset(chiv, 0, sizeof(chiv));
            memset(count, 0, sizeof(count));
            for (uint i = 0;i< num_examples ;++i) {
                // learning rate
                double eta;

                t ++;
                switch (eta_rule_type) {
                    case 0: eta = 1 / (lambda * t); break;
                    case 1: eta = 2 / (lambda * (t+1)); break;
                    default: eta = 1/(lambda*t);
                } 

                // choose random example
                uint r = get_sample(p);

                // calculate prediction
                double prediction = W * Dataset[r];

                // calculate loss
                cur_loss = max(0.0, 1 - Labels[r]*prediction);

                if(num_examples - i < 6) {
                    WeightVector old_W(dimension);
                    double pred;
                    double loss;

                    for (uint j = 0;j < num_examples;j ++) {
                        old_W = W;
                        old_W.scale(lambda);
                        pred = old_W * Dataset[j];
                        loss = max(0.0, 1- Labels[j] * pred);
                        if(loss > 0.0) {
                            old_W.add(Dataset[j], -Labels[j]);
                            count[j] ++;
                        }
                        double temp = sqrt(old_W.snorm());
                        if (temp > chiv[j]) chiv[j] = temp;
                    }
                }

                // scale w 
                W.scale(1.0 - eta*lambda/num_examples/p[r]);

                // and add to the gradient
                if (cur_loss > 0.0) {
                    double grad_weights = eta*Labels[r]/num_examples/p[r];
                    // and add sub-gradients
                    W.add(Dataset[r],grad_weights);
                }
                weight_W.add(W, t);
            }

            // update timeline
            endTime = get_runtime();
            train_time = endTime - startTime;
            startTime = get_runtime();

            WeightVector eval_W(dimension);
            eval_W.scale(0);
            if(eta_rule_type == 1) {
                eval_W = weight_W;
                eval_W.scale(2.0/t/(t+1));
            }
            else {
                eval_W = W;
            }
            // Calculate objective value
            norm_value = eval_W.snorm();
            obj_value = norm_value * lambda / 2.0;
            loss_value = 0.0;
            zero_one_error = 0.0;
            for (uint i=0; i < Dataset.size(); ++i) {
                double cur_loss = 1 - Labels[i]*(eval_W * Dataset[i]); 
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
                double cur_loss = 1 - testLabels[i]*(eval_W * testDataset[i]); 
                if (cur_loss < 0.0) cur_loss = 0.0;
                test_loss += cur_loss;
                if (cur_loss >= 1.0) test_error += 1.0;
            }
            if (testDataset.size() != 0) {
                test_loss /= testDataset.size();
                test_error /= testDataset.size();
            }

            obj[epoch] += obj_value;
            test[epoch] += test_error;

            if(change) {
                double sumup = 0;
                for(uint j=0;j<num_examples;j++) {
                    if(count[j]>0) {
                        if(p[j] == 0) {
                            p[j] = sqrt(Dataset[j].snorm());
                        }
                        else
                            p[j] = chiv[j];
                    }
                    else {
                        p[j] = 0;
                    }
                    chiv[j] = 0;
                    count[j] = 0;
                }
                for(uint j=0;j<num_examples;j++) {
                    sumup += p[j];
                }
                for(uint j=0;j<num_examples;j++) {
                    p[j] /= sumup; 
                }
            }
        }
    }

    std::cout << "SGD: " << std::endl;
    for(uint epoch = 0; epoch < num_epoch; epoch ++) {
            std::cout << "epoch #: " << epoch << endl;
            std::cout << "eta_rule_type: " << eta_rule_type << endl;
            std::cout << obj[epoch]/num_round<< " = primal objective of solution\n" 
                << test[epoch]/num_round << " = avg zero-one error over test\n" 	    
                <<  std::endl;
    }

}

void Model::localSDCA(
        std::vector<simple_sparse_vector> Dataset,
        std::vector<int> Labels,
        uint dimension,
        std::vector<simple_sparse_vector> testDataset,
        std::vector<int> testLabels,
        double lambda,
        std::vector<double> &p,
        bool change, 
        // Output variables
        long& train_time,long& calc_obj_time,
        double& obj_value,double& norm_value,
        double& loss_value,double& zero_one_error,
        double& test_loss,double& test_error,const uint num_round, const uint num_epoch) {

    uint num_examples = Labels.size();

    long startTime = get_runtime();
    long endTime;

    double chiv[num_examples];
    double count[num_examples];
    double obj[num_epoch];
    double test[num_epoch];
    double t;
    double alpha[num_examples];

    memset(obj, 0, sizeof(obj));
    memset(test, 0, sizeof(test));
    // ---------------- Main Loop -------------------
    //max_iter = num_examples;
    cout << "num_examples: " << num_examples << endl;
    for(uint round = 1; round <= num_round; round++) {
        WeightVector W(dimension);
        W.scale(0);
        memset(alpha, 0, sizeof(alpha));
        t = 0;
        for(uint epoch = 0;epoch < num_epoch; epoch++) {
            memset(chiv, 0, sizeof(chiv));
            memset(count, 0, sizeof(count));
            for (uint i = 0;i< num_examples ;++i) {
                t ++;

                // choose random example
                uint r = get_sample(p);

                // compute hinge loss gradient
                double prediction = W * Dataset[r];
                double grad = (Labels[r]*prediction  - 1.0)*(lambda*num_examples);
                // compute projected gradient
                double proj_grad = grad;
                if (alpha[r] <= 0.0)
                    proj_grad = min(grad,0.0);
                else if (alpha[r] >= 1.0)
                    proj_grad = max(grad,0.0);

                if (fabs(proj_grad) != 0.0 ) {
                    double qii  = Dataset[r].snorm();
                    double newAlpha = 1.0;
                    if (qii != 0.0) {
                        newAlpha = min(max(alpha[r] - grad / qii, 0.0), 1.0);
                    }

                    // update primal and dual variables
                    WeightVector update(dimension);
                    update.scale(0);
                    update.add(Dataset[r], 1);
                    update.scale( Labels[r]*(newAlpha-alpha[r])/(lambda*num_examples));
                    W.add(update, 1);
                    alpha[r] = newAlpha;
                }

                if(num_examples - i < 6) {
                    WeightVector old_W(dimension);
                    double pred;
                    double loss;

                    for (uint j = 0;j < num_examples;j ++) {
                        old_W = W;
                        //old_W.scale(lambda);
                        pred = old_W * Dataset[j];
                        loss = max(0.0, 1- Labels[j] * pred);
                        if(loss>0) count[j]++;
                        loss = loss + pred * alpha[j] - alpha[j];
                        if (loss > chiv[j]) chiv[j] = loss;
                    }
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

            obj[epoch] += obj_value;
            test[epoch] += test_error;

            double sumup = 0;
            for(uint j=0;j<num_examples;j++) {
                sumup += chiv[j];
            }
            cout << epoch << ": " << sumup << endl;
            /* 
            if(change) {
                double sumup = 0;
                for(uint j=0;j<num_examples;j++) {
                    p[j] = chiv[j];
                }
                for(uint j=0;j<num_examples;j++) {
                    sumup += chiv[j];
                }
<<<<<<< HEAD
                cout << epoch << ": " << sumup << endl;
=======
                cout << sumup << endl;
>>>>>>> 20855374bef68076156137b3b2fb718f5a97b957
                for(uint j=0;j<num_examples;j++) {
                    p[j] /= sumup; 
                }
            }
            */
        }
    }

    std::cout << "SDCA: " << std::endl;
    for(uint epoch = 0; epoch < num_epoch; epoch ++) {
            std::cout << "epoch #: " << epoch << std::endl;
            std::cout << obj[epoch]/num_round << std::endl;//<< " = primal objective of solution\n" 
            std::cout << test[epoch]/num_round << std::endl;// << " = avg zero-one error over test\n" 	    
            std::cout << std::endl;
    }

}

/**
void Model::SDCALearn(
        std::vector<simple_sparse_vector> Dataset,
        std::vector<int> Labels,
        uint dimension,
        std::vector<simple_sparse_vector> testDataset,
        std::vector<int> testLabels,
        double lambda,
        std::vector<double> &p,
        bool change, 
        // Output variables
        long& train_time,long& calc_obj_time,
        double& obj_value,double& norm_value,
        double& loss_value,double& zero_one_error,
        double& test_loss,double& test_error,const uint num_round, const uint num_epoch) {

    uint num_examples = Labels.size();

    long startTime = get_runtime();
    long endTime;

    double chiv[num_examples];
    double count[num_examples];
    double obj[num_epoch];
    double test[num_epoch];
    double t;
    double delta_alpha;
    double alpha[num_examples];
    double s;

    WeightVector W(dimension);
    memset(obj, 0, sizeof(obj));
    memset(test, 0, sizeof(test));
    // ---------------- Main Loop -------------------
    //max_iter = num_examples;
    s = lambda / (lambda * num_examples + 1);
    cout << "num_examples: " << num_examples << endl;
    for(uint round = 1; round <= num_round; round++) {
        W.scale(0);
        memset(alpha, 0, sizeof(alpha));
        t = 0;
        for(uint epoch = 0;epoch < num_epoch; epoch++) {
            memset(chiv, 0, sizeof(chiv));
            memset(count, 0, sizeof(count));
            for (uint i = 0;i< num_examples ;++i) {
                t ++;

                // choose random example
                uint r = get_sample(p);

                // calculate prediction
                double prediction = W * Dataset[r];

                // calculate Delta \alpha
                delta_alpha = max(-s / p[r], min(1-s / p[r], (1-Labels[r]*prediction)/Dataset[r].snorm()*lambda*num_examples))*Labels[r];

                alpha[r] += delta_alpha;
                W.add(Dataset[r], delta_alpha/lambda/num_examples);

                if(num_examples - i < 6) {
                    *
                    WeightVector old_W(dimension);
                    double pred;
                    double loss;

                    for (uint j = 0;j < num_examples;j ++) {
                        old_W = W;
                        old_W.scale(lambda);
                        pred = old_W * Dataset[j];
                        loss = max(0, 1- Labels[j] * pred);
                        if(loss > 0.0) {
                            old_W.add(Dataset[j], -Labels[j]);
                            count[j] ++;
                        }
                        double temp = sqrt(old_W.snorm());
                        if (temp > chiv[j]) chiv[j] = temp;
                    }
                    *
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

            obj[epoch] += obj_value;
            test[epoch] += test_error;

            *
            if(change) {
                double sumup = 0;
                for(uint j=0;j<num_examples;j++) {
                    if(count[j]>0) {
                        if(p[j] == 0) {
                            p[j] = sqrt(Dataset[j].snorm()) + sqrt(lambda);
                        }
                        else
                            p[j] = chiv[j];
                    }
                    else {
                        p[j] = 0;
                    }
                    chiv[j] = 0;
                    count[j] = 0;
                }
                for(uint j=0;j<num_examples;j++) {
                    sumup += p[j];
                }
                for(uint j=0;j<num_examples;j++) {
                    p[j] /= sumup; 
                }
            }
            *
        }
    }

    std::cout << "SDCA: " << std::endl;
    for(uint epoch = 0; epoch < num_epoch; epoch ++) {
            std::cout << "epoch #: " << epoch << endl;
            std::cout << obj[epoch]/num_round<< " = primal objective of solution\n" 
                << test[epoch]/num_round << " = avg zero-one error over test\n" 	    
                <<  std::endl;
    }

}
*/

// ------------------------------------------------------------//
// ---------------- READING DATA ------------------------------//
// ------------------------------------------------------------//
void Model::ReadData(// input
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
        if(label==0) label=-1;
        Labels.push_back(label);
        simple_sparse_vector instance(is,n);
        Dataset.push_back(instance);
        num_examples++;
        uint cur_max_ind = instance.max_index() + 1;
        if (cur_max_ind > dimension) dimension = cur_max_ind;
    }

    data_file.close();

    // update timeline
    readingTime = get_runtime() - startTime;

}

