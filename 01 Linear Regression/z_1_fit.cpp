/*
===========================================

C:\Users\RUI\OneDrive - 國立陽明交通大學\文件\RUI\School\NCTUECE\2_Course Data\112_2\ML\Homework\1_LinearRegression

Compile:
g++ -std=c++11 -o main main.cpp regression.cpp dataset.cpp matrix.cpp -I C:/toolbox/eigen-3.4.0

Execute (for different functions):
./main
./train
./test
===========================================
*/

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "regression.h"
#include "matrix.h"

using namespace std;

int main()
{
    LinearRegression model;
    matrix _x, _wML, X;
    int k = 3, K = 11, step = 1000;
    double M, J;
    string fn;
    vector<double>u;
    // x
    _x.range(-3, 3, step); // this should contains about 99.7% of input range since the data for trainning had been normalized
    // get
    for (int m = 5; m <= 30; m += 5)
    {
        M = (double)m;
        model.setting(m);
        model.load(m);
        u = model.u;
        // _wML^T
        _wML.resize(1, m);
        for (int j = 0; j < m; j++)
        {
            _wML[0][j] = model.wML[K * j + k - 1][0];
        }
        // phi(x)^T
        X.resize(m, step);
        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < step; i++)
            {
                X[j][i] = model.basisFunction((double)_x[0][i], k, j, m, 0.1, u[j]);
            }
        }
        // dot
        _wML.dot(X);
        fn = "homework/1_M" + to_string(m) + ".csv";
        _wML.save(fn);
    }
    return 0;
}