#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <string>
class matrix
{
public:
    matrix(int rows, int cols);
    matrix();
    ~matrix();
    // Element access
    double &at(int row, int col);
    const double &at(int row, int col) const;
    std::vector<double> &operator[](int row);
    const std::vector<double> &operator[](int row) const;
    // Operator overloads
    matrix &operator=(const matrix &rhs);
    matrix operator+(const matrix &rhs) const;
    matrix operator*(const matrix &rhs) const;
    matrix operator*(double scalar) const;
    matrix operator*(float scalar) const;
    matrix operator*(int scalar) const;
    // get property
    bool empty();
    int row() const; // #row = col size
    int col() const; // #col = row size
    std::string dim() const;
    double det();
    void resize(int newRows, int newCols, double fillValue = 0.0);
    // append data
    void append(const std::vector<double> &newRow);    // append new row
    void append(const matrix &source, int r1, int r2); // append new row
    void append(const matrix &source);                 // append new row
    void concat(const matrix &source, int c1, int c2);
    void concat(const matrix &source);
    void calcMean();
    void calcSTD(); // would calculate the mean
    void update();
    void copy(const matrix &source);
    void clear();
    // manipulation
    void I(const int d, const double val);
    void I(const int d);
    void range(const double start, const double stop, const int step);
    void T(); // transpose
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
    void abs();
    void svd(matrix &U, matrix &Sigma, matrix &V) const;
    // show
    void printRow(int ri, int l);
    void printRow(int ri);
    void print();
    void stat(); // mean & sd
    // save
    void read(const std::string &filename);
    void save(const std::string &filename) const;
    // data
    std::vector<std::vector<double>> data;
    std::vector<double> Mean, STD;
    std::vector<std::string> header;
    
private:
    int r, c;
    double Det;
};

#endif // MATRIX_H
