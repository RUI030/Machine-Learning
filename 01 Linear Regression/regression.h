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
    void eval(dataset &ds); //if valid.y_predict is empty, predict first
    void eval(); //if valid.y_predict is empty, predict first
    // read & save
    void load(const std::string& filename);
    void save(const std::string& filename);
    // member
    int M;
    int s; 
    std::vector<double> u;
    matrix PHI, wML;
    // dataset
    dataset train, valid;
    std::vector<double> mean, sd;   // for new data
};
#endif