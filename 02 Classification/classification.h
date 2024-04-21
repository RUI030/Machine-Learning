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
    void relabel(dataset &ds, int source, int target);
    void relabel(int source, int target);
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

#endif // CLASSIFICATION_H