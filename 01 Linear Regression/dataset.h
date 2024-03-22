#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include <iostream>

#include "matrix.h"

class dataset {
public:
    // Constructor and Destructor
    dataset();
    ~dataset();
    void read(const std::string& filename, int y_dim);
    void read(const std::string& filename); //default ydim=1
    void printRow(int i);
    void print();
    void copy(const dataset &source, int r1,int r2);
    void copy(const dataset &source);
    void split(dataset &train, dataset &valid, int idx);
    void T();
    void update();
    void norm();
    void normby(const dataset &source);
    void designMatrix(const int M, const double s, const std::vector<double> &u);
    void save(const std::string& filename); 
    // data
    matrix x,y,y_predict;
    // basis function
    matrix PHI;
    // Variables
    int k,xdim,ydim; // k = #feature = 11 + 1 = xdim + ydim
    int n; // # all data = train + valid = 10000 + 5818
    std::vector<double>accuracy;
};

#endif // DATASET_H
