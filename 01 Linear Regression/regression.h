#ifndef REGRESSION_H
#define REGRESSION_H

#include "dataset.h"
#include "matrix.h"
#include <vector>


class LinearRegression
{
    LinearRegression(int m);
    LinearRegression();
    ~LinearRegression();

    double basisFunction(const int k,const int j, const int xk);
    double basisFunction(const int k,const int j, matrix&);
    void update(matrix &train);
    void predict(std::vector<double>input);

    int s,M;
    std::vector<int> u;
    matrix phi;
};
#endif