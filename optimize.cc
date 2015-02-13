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
    double rand_num = rand() * p[0] / RAND_MAX;
    for (uint index = 1; index < p.size(); ++index) {
        if (rand_num < p[index]) {
            return index - 1;
        } else {
            rand_num -= p[index];
        }
    }
    return p.size()-2;
}

// SGD main algorithm
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
        bool online, bool use_ada_grad,
       // additional parameters
        int eta_rule_type, const uint& num_round, const uint& num_epoch) {
    uint num_examples = Labels.size();

    double t;
    std::vector<double> prob;
    std::deque<double> blank;
    WeightVector W(dimension);
    WeightVector rW(dimension);
    WeightVector C(dimension);
    WeightVector weight_W(dimension);
    WeightVector old_W(dimension);
    WeightVector old_W2(dimension);
    WeightVector eval_W(dimension);
    WeightVector G(dimension);

    output.resize(num_epoch);
    for (std::vector<ResultStruct>::iterator out = output.begin(); out != output.end(); out++ ) { 
        out->total_time = 0;
        out->train_time = 0;
        out->calc_obj_time = 0;
        out->norm_value = 0;
        out->loss_value = 0;
        out->zero_one_error = 0;
        out->obj_value = 0;
        out->test_loss = 0;
        out->test_error = 0;
    }

    // ---------------- Main Loop -------------------
    for (uint round = 1; round <= num_round; round++) {
        W.scale(0);
        G.scale(0);
        weight_W.scale(0);
        prob = p;
        t = 0;
        if (online) {
            recent.clear();
            for (uint i = 0; i < num_examples; ++i) {
                recent.push_back(blank);
                recent[i].push_back(sqrt(Dataset[i].snorm()) + sqrt(lambda));
            }
        }

        for (uint epoch = 0; epoch < num_epoch; epoch++) {
           if (is_adaptive) {
                std::fill(chiv.begin(), chiv.end(), 0);
                std::fill(count.begin(), count.end(), 0);
            }
            if (use_variance_reduction && epoch > 0) {
                rW = W;
                C.scale(0);
                for (uint i = 0; i < num_examples; ++i) {
                    precompute[i] = W * Dataset[i];
                    double loss = std::max(0.0, 1.0 - Labels[i] * precompute[i]);
                    if (loss > 0.0) {
                        C.add(Dataset[i], -Labels[i]);
                    }
                }
                C.scale(1.0 / num_examples);
                C.add(W, lambda);
            }

            double epoch_start_time = GetRuntime();
            double train_startTime = GetRuntime();
 
            // learning rate
            double eta;
            double prediction;
            double cur_loss;

            for (uint i = 0; i < num_examples; ++i) {
                ++t;

                if (eta_rule_type == 0) {
                    eta = 1.0 / (lambda * t); 
                } else {
                    eta = 2.0 / (lambda * (t+1));
                }

                // choose random example
                uint r = GetSample(prob);

                // calculate prediction
                prediction = W * Dataset[r];

                // calculate loss
                cur_loss = std::max(0.0, 1.0 - Labels[r] * prediction);

                if (online) {
                    double temp = W.snorm() * lambda * lambda;
                    if(cur_loss > 0.0) {
                        temp += Dataset[r].snorm() * Labels[r] * Labels[r] - prediction * Labels[r] * lambda * 2.0;
                    }
                    temp = sqrt(temp);
                    if(recent[r].size()==100) recent[r].pop_front();
                    recent[r].push_back(temp);
                    double peek = 0;
                    for(std::deque<double>::iterator ele = recent[r].begin(); ele!=recent[r].end();ele++) 
                        if (*ele > peek) peek = *ele;
                    prob[0] += peek - prob[r + 1];
                    prob[r + 1] = peek; 
                } 
                else {
                    if (is_adaptive && num_examples - i < 6) {
                        double pred;
                        double loss;
                        double temp;

                        for (uint j = 0; j < num_examples; ++j) {
                            pred = W * Dataset[j];
                            loss = std::max(0.0, 1.0 - Labels[j] * pred);
                            temp = W.snorm() * lambda * lambda;
                            if (loss > 0.0) {
                                temp = temp + Dataset[j].snorm() * Labels[j] * Labels[j] - pred * Labels[j] * lambda * 2.0;
                                ++ count[j];
                            }
                            temp = sqrt(temp);
                            if (temp > chiv[j]) chiv[j] = temp;
                        }
                    }
                }
                if (use_variance_reduction && epoch > 0) {
                    old_W = W;
                    old_W.scale(lambda);
                    if (cur_loss > 0.0) {
                        old_W.add(Dataset[r], -Labels[r]);
                    }
                    old_W2 = rW;
                    old_W2.scale(lambda);
                    double loss = std::max(0.0, 1.0 - Labels[r] * precompute[r]);
                    if (loss > 0.0) {
                        old_W2.add(Dataset[r], -Labels[r]);
                    }
                    old_W.add(old_W2, -1);
                    old_W.add(C, num_examples * prob[r + 1] / prob[0]);
                    W.add(old_W, -eta / (num_examples * prob[r + 1] / prob[0]));
                } else {
                    if (use_ada_grad) {
                        old_W = W;
                        old_W.scale(lambda);
                        if (cur_loss > 0.0) 
                            old_W.add(Dataset[r], -Labels[r]);
                        old_W.scale(num_examples * prob[r + 1] / prob[0]);
                        G.sqr_add(old_W);
                        old_W.pair_mul(G);
                        W.add(old_W, -eta);
                    } else {
                        W.scale(1.0 - lambda * eta / (num_examples * prob[r + 1] / prob[0]));
                        if(cur_loss > 0.0) {
                            W.add(Dataset[r], Labels[r] * eta / (num_examples * prob[r + 1] / prob[0]));
                        }
                    }
                }
                if (eta_rule_type == 1) {
                    weight_W.add(W, t);
                }
            }

            // update timeline
            double train_endTime = GetRuntime();
            double train_time = train_endTime - train_startTime;
            double calc_obj_startTime = GetRuntime();

            if (eta_rule_type == 1) {
                eval_W = W;
                W = weight_W;
                W.scale(2.0/t/(t+1));
            }
            // Calculate objective value
            double norm_value = W.snorm();
            double obj_value = norm_value * lambda / 2.0;
            double loss_value = 0.0;
            double zero_one_error = 0.0;
            for (uint i=0; i < Dataset.size(); ++i) {
                double cur_loss = 1 - Labels[i] * (W * Dataset[i]);
                if (cur_loss < 0.0) cur_loss = 0.0;
                loss_value += cur_loss / num_examples;
                obj_value += cur_loss / num_examples;
                if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
            }

            double calc_obj_endTime = GetRuntime();
            double calc_obj_time = calc_obj_endTime - calc_obj_startTime;

            // Calculate test_loss and test_error
            double test_loss = 0.0;
            double test_error = 0.0;
            for (uint i=0; i < testDataset.size(); ++i) {
                double cur_loss = 1 - testLabels[i] * (W * testDataset[i]);
                if (cur_loss < 0.0) cur_loss = 0.0;
                test_loss += cur_loss;
                if (cur_loss >= 1.0) test_error += 1.0;
            }
            if (testDataset.size() != 0) {
                test_loss /= testDataset.size();
                test_error /= testDataset.size();
            }

            if (eta_rule_type == 1) {
                W = eval_W;
            }

            output[epoch].train_time += train_time;
            output[epoch].calc_obj_time += calc_obj_time;
            output[epoch].norm_value += norm_value;
            output[epoch].loss_value += loss_value;
            output[epoch].zero_one_error += zero_one_error;
            output[epoch].obj_value += obj_value;
            output[epoch].test_loss += test_loss;
            output[epoch].test_error += test_error;

            if (!online && is_adaptive) {
                double sumup = 0;
                double comeup = 0;
                for (uint j = 0; j < num_examples; ++j) {
                    if (count[j] > 0) {
                        if (prob[j + 1] / prob[0]  < 0.1 / num_examples) {
                            prob[j + 1] = p[j + 1];
                        } else {
                            prob[j + 1] = chiv[j];
                        }
                    } else {
                        prob[j + 1] = -1; 
                    }
                    chiv[j] = 0;
                    count[j] = 0;
                }
                for (uint j = 1; j <= num_examples; ++j) {
                    if(prob[j]>0) sumup += prob[j]; else comeup ++;
                }
                for (uint j = 1; j <= num_examples; ++j) {
                    if(prob[j]>0) prob[j] /= (sumup+comeup); else prob[j] = 1.0 / (sumup+comeup); 
                }
                prob[0] = 1;
            }
            double epoch_end_time= GetRuntime();
            output[epoch].total_time += epoch_end_time - epoch_start_time;
        }
    }

    for (std::vector<ResultStruct>::iterator out = output.begin(); out != output.end(); out++ ) { 
        out->train_time /= num_round;
        out->calc_obj_time /= num_round;
        out->norm_value /= num_round;
        out->loss_value /= num_round;
        out->zero_one_error /= num_round;
        out->obj_value /= num_round;
        out->test_loss /= num_round;
        out->test_error /= num_round;
        out->total_time /= num_round;
    }

    // W.print(std::cout);
    Print();
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
        bool ada_rule_type, 
        bool online,
        // Additional parameters
        const uint &num_round, const uint &num_epoch ) {
    uint num_examples = Labels.size();

    double t;
    std::vector<double> prob;
    std::deque<double> blank;
    WeightVector W(dimension);

    output.resize(num_epoch);
    for (std::vector<ResultStruct>::iterator out = output.begin(); out != output.end(); out++ ) { 
        out->total_time = 0;
        out->train_time = 0;
        out->calc_obj_time = 0;
        out->norm_value = 0;
        out->loss_value = 0;
        out->zero_one_error = 0;
        out->obj_value = 0;
        out->test_loss = 0;
        out->test_error = 0;
    }

    // ---------------- Main Loop -------------------
    for (uint round = 1; round <= num_round; round++) {
        W.scale(0);
        std::fill(alpha.begin(), alpha.end(), 0);
        prob = p;
        t = 0;
        if(online) {
            recent.clear();
            for (uint i = 0;i < num_examples; ++i) {
                recent.push_back(blank);
                recent[i].push_back(sqrt(Dataset[i].snorm()) + sqrt(lambda));
            }
        }
        for (uint epoch = 0; epoch < num_epoch; epoch++) {
            
            /*
             * print out probability
            for (uint i = 1; i <= num_examples; ++i) {
                std::cout << 1.0 * prob[i] / prob[0] << " \n"[i == num_examples];
            }
            */

            // -----------------
            double train_startTime = GetRuntime();
            double epoch_start_time = GetRuntime();
            double sample_time = 0;
            if (is_adaptive) {
                std::fill(chiv.begin(), chiv.end(), 0);
                std::fill(count.begin(), count.end(), 0);
            }
            for (uint i = 0; i < num_examples; ++i) {
                ++t;

                // choose random example
                double sample_start = GetRuntime();
                uint r = GetSample(prob);
                sample_time += GetRuntime() - sample_start;

                // calculate prediction
                double prediction = W * Dataset[r];

                // calculate Delta \alpha
                double delta_alpha = std::max(0.0, std::min(1.0,
                        (1.0 - Labels[r] * prediction) / Dataset[r].snorm()
                        * lambda * num_examples + alpha[r] * Labels[r]))
                    * Labels[r] - alpha[r];

                if (online) {
                    double loss = std::max(0.0, 1 - Labels[r] * prediction);
                    double temp = 0;
                    if (ada_rule_type == 0) {
                        loss = loss + alpha[r] * (prediction - Labels[r]);
                        temp = loss;
                    } else {
                        temp = W.snorm() * lambda * lambda;
                        if (loss > 0.0) {
                            temp += Dataset[r].snorm() * Labels[r] * Labels[r] - prediction * Labels[r] * lambda * 2.0;
                        }
                        temp = sqrt(temp);
                    }
                    if (recent[r].size()==1000) recent[r].pop_front();
                    recent[r].push_back(temp);
                    double peek = 0;
                    for(std::deque<double>::iterator ele = recent[r].begin(); ele != recent[r].end(); ele++) 
                        if (*ele > peek) peek = *ele;
                    prob[0] += peek - prob[r + 1];
                    prob[r + 1] = peek;
                }

                alpha[r] += delta_alpha;
                W.add(Dataset[r], delta_alpha / lambda / num_examples);


                if (!online && is_adaptive && num_examples - i < 6) {
                    if (ada_rule_type == 1) {
                        double pred;
                        double loss;
                        double temp;

                        for (uint j = 0; j < num_examples; ++j) {
                            pred = W * Dataset[j];
                            loss = std::max(0.0, 1.0 - Labels[j] * pred);
                            temp = W.snorm() * lambda * lambda;
                            if (loss > 0.0) {
                                temp = temp + Dataset[j].snorm() * Labels[j] * Labels[j] - pred * Labels[j] * lambda * 2.0;
                                ++count[j];
                            }
                            temp = sqrt(temp);
                            if (temp > chiv[j]) chiv[j] = temp;
                        }
                    } else {
                        double pred;
                        double loss;
                        double sumup = 0;

                        for (uint j = 0; j < num_examples; j ++) {
                            pred = W * Dataset[j];
                            loss = std::max(0.0, 1- Labels[j] * pred);
                            if (loss > 0) count[j]++;
                            loss = loss + alpha[j]*(pred-Labels[j]);
                            sumup += loss;
                            if (loss > chiv[j]) chiv[j] = loss;
                        }
                        sumup /= num_examples;
                        //std::cout << "epoch:" << epoch << " " << sumup << std::endl;
                    }
                }

            }

            // update timeline
            double train_endTime = GetRuntime() - sample_time;
            double train_time = train_endTime - train_startTime;
            double calc_obj_startTime = GetRuntime();

            // Calculate objective value
            double norm_value = W.snorm();
            double obj_value = norm_value * lambda / 2.0;
            double loss_value = 0.0;
            double zero_one_error = 0.0;
            for (uint i = 0; i < Dataset.size(); ++i) {
                double cur_loss = 1 - Labels[i]*(W * Dataset[i]);
                if (cur_loss < 0.0) cur_loss = 0.0;
                loss_value += cur_loss/num_examples;
                obj_value += cur_loss/num_examples;
                if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
            }

            double calc_obj_endTime = GetRuntime();
            double calc_obj_time = calc_obj_endTime - calc_obj_startTime;

            // Calculate test_loss and test_error
            double test_loss = 0.0;
            double test_error = 0.0;
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
            
            output[epoch].train_time += train_time;
            output[epoch].calc_obj_time += calc_obj_time;
            output[epoch].norm_value += norm_value;
            output[epoch].loss_value += loss_value;
            output[epoch].zero_one_error += zero_one_error;
            output[epoch].obj_value += obj_value;
            output[epoch].test_loss += test_loss;
            output[epoch].test_error += test_error;

            if (is_adaptive && !online) {
                if (ada_rule_type == 0) {
                    double sumup = 0;
                    double comeup = 0;
                    for (uint j = 0; j < num_examples; ++j) {
                        if (count[j] > 0) {
                            if (prob[j + 1] / prob[0]  < 0.1 / num_examples) {
                                prob[j + 1] = p[j + 1];
                            } else {
                                prob[j + 1] = chiv[j];
                            }
                        } else {
                            prob[j + 1] = -1; 
                        }
                        chiv[j] = 0;
                        count[j] = 0;
                    }
                    for (uint j = 1; j <= num_examples; ++j) {
                        if(prob[j]>0) sumup += prob[j]; else comeup ++;
                    }
                    for (uint j = 1; j <= num_examples; ++j) {
                        if(prob[j]>0) prob[j] /= (sumup+comeup); else prob[j] = 1.0 / (sumup+comeup); 
                    }
                    prob[0] = 1;

                    /*
                    for (uint j = 0; j < num_examples; ++j) {
                        prob[0] -= prob[j + 1];
                        if (count[j] > 0) {
                            if (prob[j + 1] < 1.01e-5) {
                                prob[j + 1] = p[j + 1];
                            } else {
                                prob[j + 1] = chiv[j];
                            }
                        } else {
                            prob[j + 1] = 1e-5;
                        }
                        chiv[j] = 0;
                        count[j] = 0;
                        prob[0] += prob[j + 1];
                    }
                    */
                } else {
                    for (uint j = 0; j < num_examples; ++j) {
                        prob[0] += chiv[j] - prob[j + 1];
                        prob[j + 1] = chiv[j];
                    }
                }
            }
            double epoch_end_time= GetRuntime();
            output[epoch].total_time += epoch_end_time - epoch_start_time;
        }
    }
    for (std::vector<ResultStruct>::iterator out = output.begin(); out != output.end(); out++ ) { 
        out->total_time /= num_round;
        out->train_time /= num_round;
        out->calc_obj_time /= num_round;
        out->norm_value /= num_round;
        out->loss_value /= num_round;
        out->zero_one_error /= num_round;
        out->obj_value /= num_round;
        out->test_loss /= num_round;
        out->test_error /= num_round;
    }
    // W.print(std::cout);
    Print();
}

void Model::Print() {
    std::cout << "Total_time:\t";
    for (auto out: output) {
        std::cout << out.total_time << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Train_time:\t";
    for (auto out: output) {
        std::cout << out.train_time << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Calc_obj_time:\t"; 
    for (auto out: output) {
        std::cout << out.calc_obj_time << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Norm_value:\t";
    for (auto out: output) {
        std::cout << out.norm_value << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Loss_value:\t";
    for (auto out: output) {
        std::cout << out.loss_value << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Zero_one_error:\t";
    for (auto out: output) {
        std::cout << out.zero_one_error << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Obj_value:\t";
    for (auto out: output) {
        std::cout << out.obj_value << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Test_loss:\t";
    for (auto out: output) {
        std::cout << out.test_loss << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Test_error:\t";
    for (auto out: output) {
        std::cout << out.test_error << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
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
    num_examples = 0;
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

    chiv.resize(num_examples);
    alpha.resize(num_examples);
    precompute.resize(num_examples);
    count.resize(num_examples);
}

