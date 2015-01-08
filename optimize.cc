/* Copyright [2015] <Raynald Chung>
 * ETH Zürich
 */

#include "include/optimize.h"
#include <algorithm>

// Help function for getting runtime
double GetRuntime(void) {
    clock_t start;
    start = clock();
    return static_cast<double>(start)/static_cast<double>(CLOCKS_PER_SEC);
}

// Function for getting an index of a sample point according to p
uint GetSample(const std::vector<double> &p) {
    double rand_num = rand() * 1.0 / RAND_MAX;
    for (uint index = 0; index < p.size(); ++index) {
        if (rand_num < p[index]) {
            return index;
        } else {
            rand_num -= p[index];
        }
    }
    return p.size()-1;
}

void Model::SGDLearn(
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
        // additional parameters
        int eta_rule_type, const uint& num_round, const uint& num_epoch) {
    uint num_examples = Labels.size();
    double startTime = GetRuntime();
    double endTime;
    double chiv[kNUMEXAMPLE];
    double count[kNUMEXAMPLE];
    double obj[kMAXEPOCH];
    double test[kMAXEPOCH];
    double t;
    double cur_loss;
    std::vector<double> prob;
    WeightVector W(dimension);
    WeightVector C(dimension);
    WeightVector rW(dimension);
    WeightVector weight_W(dimension);
    WeightVector old_W(dimension);
    WeightVector old_W2(dimension);
    memset(obj, 0, sizeof(obj));
    memset(test, 0, sizeof(test));
    // ---------------- Main Loop -------------------
    std::cout << "num_examples: " << num_examples << std::endl;
    for (uint round = 1; round <= num_round; round++) {
        W.scale(0);
        weight_W.scale(0);
        prob = p;
        t = 0;
        for (uint epoch = 0; epoch < num_epoch; epoch++) {
            memset(count, 0, sizeof(count));
            memset(chiv, 0, sizeof(chiv));
            if (use_variance_reduction && epoch > 0) {
                rW = W;
                C.scale(0);
                for (uint i = 0; i < num_examples; ++i) {
                    double pred = rW * Dataset[i];
                    double loss = std::max(0.0, 1 - Labels[i] * pred);
                    if (loss > 0.0) {
                        C.add(Dataset[i], -Labels[i]);
                    }
                }
                C.scale(1.0 / num_examples);
                C.add(W, lambda);
            }
            for (uint i = 0; i < num_examples; ++i) {
                // learning rate
                double eta;

                ++t;
                switch (eta_rule_type) {
                    case 0: eta = 1 / (lambda * t); break;
                    case 1: eta = 2 / (lambda * (t+1)); break;
                    default: eta = 1/(lambda*t);
                }

                // choose random example
                uint r = GetSample(prob);

                // calculate prediction
                double prediction = W * Dataset[r];

                // calculate loss
                cur_loss = std::max(0.0, 1 - Labels[r]*prediction);

                if (num_examples - i < 20) {
                    double pred;
                    double loss;

                    for (uint j = 0; j < num_examples; ++j) {
                        old_W = W;
                        old_W.scale(lambda);
                        pred = old_W * Dataset[j];
                        loss = std::max(0.0, 1- Labels[j] * pred);
                        if (loss > 0.0) {
                            old_W.add(Dataset[j], -Labels[j]);
                            ++count[j];
                        }
                        double temp = sqrt(old_W.snorm());
                        if (temp > chiv[j]) chiv[j] = temp;
                        // chiv[j] += temp;
                    }
                }

                old_W = W;
                old_W.scale(lambda);
                if (cur_loss > 0.0) {
                    old_W.add(Dataset[r], -Labels[r]);
                }

                if (use_variance_reduction && epoch > 0) {
                    old_W2 = rW;
                    old_W2.scale(lambda);
                    double pred = rW * Dataset[r];
                    double loss = std::max(0.0, 1 - Labels[r] * pred);
                    if (loss > 0.0) {
                        old_W2.add(Dataset[r], -Labels[r]);
                    }
                    old_W.add(old_W2, -1);
                    old_W.add(C, num_examples * prob[r]);
                }
                W.add(old_W, -eta / (num_examples * prob[r]));
                weight_W.add(W, t);
            }

            // update timeline
            endTime = GetRuntime();
            train_time = endTime - startTime;
            startTime = GetRuntime();

            WeightVector eval_W(dimension);
            eval_W.scale(0);
            if (eta_rule_type == 1) {
                eval_W = weight_W;
                eval_W.scale(2.0/t/(t+1));
            } else {
                eval_W = W;
            }
            // Calculate objective value
            norm_value = eval_W.snorm();
            obj_value = norm_value * lambda / 2.0;
            loss_value = 0.0;
            zero_one_error = 0.0;
            for (uint i=0; i < Dataset.size(); ++i) {
                double cur_loss = 1 - Labels[i] * (eval_W * Dataset[i]);
                if (cur_loss < 0.0) cur_loss = 0.0;
                loss_value += cur_loss / num_examples;
                obj_value += cur_loss / num_examples;
                if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
            }

            endTime = GetRuntime();
            calc_obj_time = endTime - startTime;

            // Calculate test_loss and test_error
            test_loss = 0.0;
            test_error = 0.0;
            for (uint i=0; i < testDataset.size(); ++i) {
                double cur_loss = 1 - testLabels[i] * (eval_W * testDataset[i]);
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

            if (is_adaptive) {
                double sumup = 0;
                for (uint j = 0; j < num_examples; ++j) {
                    if (count[j] > 0) {
                        if (prob[j] <= 1.1 / num_examples / 1e3) {
                            prob[j] = std::max(chiv[j], sqrt(Dataset[j].snorm()));
                        } else {
                            prob[j] = std::max(chiv[j], 1.0 / num_examples / 1e3);
                        }
                    } else {
                        prob[j] = 1.0 / num_examples / 1e3;
                    }
                    chiv[j] = 0;
                    count[j] = 0;
                }
                for (uint j = 0; j < num_examples; ++j) {
                    sumup += prob[j];
                }
                for (uint j = 0; j < num_examples; ++j) {
                    prob[j] /= sumup;
                }
            }
        }
    }

    std::cout << "SGD: " << std::endl;
    std::cout << " = primal objective of solution\n";
    for (uint epoch = 0; epoch < num_epoch; ++epoch) {
        std::cout << obj[epoch] / num_round << " ";
    }
    std::cout << std::endl;
    std::cout << " = avg zero-one error over test\n";
    for (uint epoch = 0; epoch < num_epoch; ++epoch) {
        std::cout << test[epoch] / num_round << " ";
    }
    std::cout << std::endl;
}

void Model::SDCALearn(
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
        const uint &num_round, const uint &num_epoch ) {
    uint num_examples = Labels.size();

    double startTime = GetRuntime();
    double endTime;

    double chiv[kNUMEXAMPLE];
    double count[kNUMEXAMPLE];
    double obj[kMAXEPOCH];
    double test[kMAXEPOCH];
    double t;
    double alpha[kNUMEXAMPLE];
    std::vector<double> prob;

    WeightVector W(dimension);
    memset(obj, 0, sizeof(obj));
    memset(test, 0, sizeof(test));
    // ---------------- Main Loop -------------------
    std::cout << "num_examples: " << num_examples << std::endl;
    for (uint round = 1; round <= num_round; round++) {
        W.scale(0);
        memset(alpha, 0, sizeof(alpha));
        prob = p;
        t = 0;
        for (uint epoch = 0; epoch < num_epoch; epoch++) {
            memset(chiv, 0, sizeof(chiv));
            memset(count, 0, sizeof(count));
            for (uint i = 0; i < num_examples; ++i) {
                ++t;

                // choose random example
                uint r = GetSample(prob);

                // calculate prediction
                double prediction = W * Dataset[r];

                // calculate Delta \alpha
                double delta_alpha = std::max(0.0, std::min(1.0,
                        (1.0 - Labels[r] * prediction) / Dataset[r].snorm()
                        * lambda * num_examples + alpha[r] * Labels[r]))
                    * Labels[r] - alpha[r];

                alpha[r] += delta_alpha;
                W.add(Dataset[r], delta_alpha / lambda / num_examples);

                if (num_examples - i < 6) {
                    WeightVector old_W(dimension);
                    double pred;
                    double loss;
                    double sumup = 0;
                    old_W = W;

                    for (uint j = 0; j < num_examples; j ++) {
                        pred = old_W * Dataset[j];
                        loss = std::max(0.0, 1- Labels[j] * pred);
                        if (loss > 0) count[j]++;
                        loss = loss + alpha[j]*(pred-Labels[j]);
                        sumup += loss;
                        if (loss > chiv[j]) chiv[j] = loss;
                    }
                    sumup /= num_examples;
                }
            }

            // update timeline
            endTime = GetRuntime();
            train_time = endTime - startTime;
            startTime = GetRuntime();

            // Calculate objective value
            norm_value = W.snorm();
            obj_value = norm_value * lambda / 2.0;
            loss_value = 0.0;
            zero_one_error = 0.0;
            for (uint i = 0; i < Dataset.size(); ++i) {
                double cur_loss = 1 - Labels[i]*(W * Dataset[i]);
                if (cur_loss < 0.0) cur_loss = 0.0;
                loss_value += cur_loss/num_examples;
                obj_value += cur_loss/num_examples;
                if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
            }

            endTime = GetRuntime();
            calc_obj_time = endTime - startTime;

            // Calculate test_loss and test_error
            test_loss = 0.0;
            test_error = 0.0;
            for (uint i = 0; i < testDataset.size(); ++i) {
                double cur_loss = 1 - testLabels[i] * (W * testDataset[i]);
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

            if (is_adaptive) {
                double sumup = 0;
                for (uint j = 0; j < num_examples; ++j) {
                    prob[j] = chiv[j];
                }
                for (uint j = 0; j < num_examples; ++j) {
                    sumup += chiv[j];
                }
                for (uint j = 0; j < num_examples; ++j) {
                    prob[j] /= sumup;
                }
            }
       }
    }

    std::cout << "SDCA: " << std::endl;
    std::cout<< " = primal objective of solution\n";
    for (uint epoch = 0; epoch < num_epoch; epoch ++) {
        std::cout << obj[epoch]/num_round << " ";
    }
    std::cout << std::endl;
    std::cout << " = avg zero-one error over test\n";
    for (uint epoch = 0; epoch < num_epoch; epoch ++) {
        std::cout << test[epoch]/num_round << " ";
    }
    std::cout << std::endl;
    /*
    std::cout << " = time \n";
    for (uint epoch = 0; epoch < num_epoch; epoch ++) {
        std::cout << test[epoch]/num_round << " ";
    }
    cout << endl;
    */
}

// ---------------- READING DATA ------------------------------//
void Model::ReadData(
        // input
        std::string& data_filename,
        // output
        std::vector<simple_sparse_vector> & Dataset,
        std::vector<int> & Labels,
        uint& dimension,
        double& readingTime) {
    dimension = 0;

    // Start a timer
    double startTime = GetRuntime();

    // OPEN DATA FILE
    std::ifstream data_file(data_filename.c_str());
    if (!data_file.good()) {
        std::cerr << "error w/ " << data_filename << std::endl;
        exit(EXIT_FAILURE);
    }


    // Read SVM-Light data file
    // ========================
    int num_examples = 0;
    std::string buf;
    while (getline(data_file, buf)) {
        // ignore lines which begin with #
        if (buf[0] == '#') continue;
        // Erase what comes after #
        size_t pos = buf.find('#');
        if (pos < buf.size()) {
            buf.erase(pos);
        }
        // replace ':' with white space
        int n = 0;
        for (size_t pos = 0; pos < buf.size(); ++pos)
            if (buf[pos] == ':') {
                n++; buf[pos] = ' ';
            }
        // read from the string
        std::istringstream is(buf);
        int label = 0;
        is >> label;
        if (label == 0 || label == 2) label = -1;
        Labels.push_back(label);
        simple_sparse_vector instance(is, n);
        Dataset.push_back(instance);
        num_examples++;
        uint cur_max_ind = instance.max_index() + 1;
        if (cur_max_ind > dimension) dimension = cur_max_ind;
    }

    data_file.close();

    // update timeline
    readingTime = GetRuntime() - startTime;
}

