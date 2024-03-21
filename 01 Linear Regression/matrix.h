#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class matrix
{
public:
    matrix();
    ~matrix();
    // get property
    int row() const; // #row = col size
    int col() const; // #col = row size
    double det();
    // append data
    void append(const std::vector<double> &newRow); // append new row
    void append(const matrix &source,int r1, int r2); // append new row
    void append(const matrix &source); // append new row
    void concat(const matrix &source, int c1, int c2);
    void concat(const matrix &source);
    void calcMean();
    void calcSTD(); // would calculate the mean
    void update();
    void copy(const matrix &source);
    void clear();
    // manipulation
    void T();         // transpose
    void normalize(const std::vector<double> m, const std::vector<double> s); 
    void normalize(const matrix &source); 
    void normalize(); 
    void slice(const matrix &source, int r1, int r2, int c1, int c2);
    void slice(const matrix &source, int r1, int r2);
    void fill(double val);
    void add(const matrix &source);
    void sub(const matrix &source);
    void dot(const matrix &source);
    void scale(double val);
    void calcDet();
    void inv();
    // show
    void printRow(int ri, int l);
    void printRow(int ri);
    void print();
    void stat();
    // save
    // data
    std::vector<std::vector<double>> data;
    std::vector<double> Mean, STD;
    std::vector<std::string> header;

private:
    int r, c;
    double Det;
};

#endif // MATRIX_H
