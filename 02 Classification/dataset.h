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
    void normalize(const std::vector<double> m, const std::vector<double> s, const std::vector<double> m_y, const std::vector<double> s_y);
    void normalize(const dataset &source);
    void normalize();
    void save(const std::string& filename); 
    void clear();
    // Regression
    void designMatrix(const int M, const double s, const std::vector<double> &u);
    // Classification
    void relabel(double source, double target);
    void ConfusionMatrix();
    // data
    matrix x,y,y_predict, y_k;
    // Variables
    int k = 4 ,xdim,ydim; // #class = 4, #feature = 2, #label = 1
    int n;
    // Regression
    matrix PHI; // design matrix
    std::vector<double>accuracy, MSE;
    // Classification
    matrix confusion_matrix;
};

#endif // DATASET_H
