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
    void prep(dataset &ds, int i);
    void prep(dataset &ds);
    void update(const matrix &X, const matrix &t);
    void update(const dataset &ds);
    void update();
    void predict(dataset &ds);
    void eval(dataset &ds, bool doNorm); //if valid.y_predict is empty, predict first
    void eval(dataset &ds); //if valid.y_predict is empty, predict first
    void eval(); //if valid.y_predict is empty, predict first
    void normalize(matrix &input);
    void normalize(dataset &input);
    void setting(const int _m, const double _s);
    void setting(const int _m);
    // read & save
    void rename(const std::string& modelName);
    void load(const std::string& modelName, const std::string& pre);
    void load(const std::string& modelName);
    void load(const int _m);
    void save(const std::string& modelName, const std::string& pre);
    void save(const std::string& modelName);
    void save();
    // member
    double M, s; 
    std::vector<double> u;
    matrix PHI, wML;
    // dataset
    dataset train, valid;
    std::vector<double> mean, sd, mean_y, sd_y;   // for new data
    std::string name;
};
#endif