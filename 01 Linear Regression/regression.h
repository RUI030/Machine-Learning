#ifndef REGRESSION_H
#define REGRESSION_H

#include "dataset.h"
#include "matrix.h"
#include <vector>

class LinearRegression
{
public:
    LinearRegression(int m);
    LinearRegression();
    ~LinearRegression();
    void split(dataset &ds, int i);
    void split(dataset &ds);
    void prep();
    double basisFunction(double val, int k, int j, int m, double S, double uj);
    double basisFunction(double val, int k, int j, int m);
    double basisFunction(double val, int k, int j);
    void update(matrix &X, const matrix &t);
    void update(dataset &ds);
    void update();
    void update2(matrix &X, const matrix &t);
    void update2(dataset &ds);
    void update2();
    void predict(dataset &ds, bool doNorm);
    void predict(dataset &ds);
    void eval(dataset &ds, bool doNorm); // if valid.y_predict is empty, predict first
    void eval(dataset &ds);              // if valid.y_predict is empty, predict first
    void eval();                         // if valid.y_predict is empty, predict first
    void normalize(matrix &input);
    void normalize(dataset &input);
    void setting(const int _m, const double _s, const double l);
    void setting(const int _m, const double _s);
    void setting(const int _m);
    // read & save
    void rename(const std::string &modelName);
    void load(const std::string &modelName, const std::string &pre);
    void load(const std::string &modelName);
    void load(const int _m);
    void save(const std::string &modelName, const std::string &pre);
    void save(const std::string &modelName);
    void save();
    // member
    double M, s, lambda;
    std::vector<double> u;
    matrix PHI, wML;
    // dataset
    dataset train, valid;
    std::vector<double> mean, sd, mean_y, sd_y; // for new data
    std::string name;
};
#endif