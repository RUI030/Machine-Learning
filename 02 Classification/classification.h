#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include "matrix.h"
#include "dataset.h"
#include <vector>

using namespace std;

class GenerativeModel
{
private:
    /* data */
public:
    GenerativeModel(/* args */);
    ~GenerativeModel();
    void relabel(double source, double target);
    void update(matrix &X, matrix &t);
    void update(dataset &ds);
    void update();
    void predict(dataset &ds);
    void eval(dataset &ds);              // if valid.y_predict is empty, predict first
    void eval();      
    // load model & save model
    void rename(const std::string &modelName);
    void load(const std::string &modelName, const std::string &pre);
    void load(const std::string &modelName);
    void save(const std::string &modelName, const std::string &pre);
    void save(const std::string &modelName);
    void save();

    // Data
    dataset train, valid;
    // statics
    int N; // #data
    vector<int> Nk; // #data for each class
    // model
    int k, nf; // #class, #features
    vector<double> pi;
    matrix w, w0; // [w1 ... wk] => a_k(x) = x dot w + w0
    matrix mu, LAMBDA; // mu[feat][class]
    std::string name;
};

class DiscriminativeModel
{   
private:
    /* data */  
public:
    DiscriminativeModel(/* args */);
    ~DiscriminativeModel();
    void relabel(double source, double target);
    void setting(double _lr, int _batch_size, int _epoch);
    void randWeight();
    void batchPredict(dataset &ds, int start, int end);
    void batchUpdate(dataset &ds, int start, int end);
    void update(double decay, int step);
    void update();
    void predict(dataset &ds);
    void eval(dataset &ds, bool showCM);              // if valid.y_predict is empty, predict first
    void eval(dataset &ds);              // if valid.y_predict is empty, predict first
    void eval();      
    // load model & save model
    void rename(const std::string &modelName);
    void load(const std::string &modelName, const std::string &pre);
    void load(const std::string &modelName);
    void save(const std::string &modelName, const std::string &pre, int ep);
    void save(const std::string &modelName, int ep);
    void save(int ep);
    void save();
    void saveLog(const std::string &filename);
    void saveLog();
    // Data
    dataset train, valid;
    // statics
    int N; // #data
    int best_ep;
    // model
    int K, M; // #class, #phi
    matrix w; // w[M][K]
    std::string name;
    matrix update_log; // loss func, train acc, valid acc
    // hyperparameters
    double lr;
    int batch_size, epoch;
};

#endif // CLASSIFICATION_H