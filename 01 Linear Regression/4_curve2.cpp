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
    dataset x3;
    int k = 2, K = 11, step = 1000;
    double M, J;
    string fn, mName;
    vector<double> u;
    double min = 0.0, max = 1.0, tmp;

    min = -4.0;
    max = 2.0;

    // x
    x3.x.resize(step, K);
    x3.y.resize(step, 1);
    x3.n = step;
    for (int i = 0; i < step; i++)
    {
        x3.y.data[i][0] = 0;
        tmp = min + ((double)i) * (max - min) / ((double)step);
        for (int j = 0; j < K; j++)
        {
            if (j == k)
                x3.x.data[i][j] = tmp;
            else
                x3.x.data[i][j] = 0.0;
        }
    }
    // x3.print();
    for (int m = 5; m <= 30; m += 5)
    {
        mName = "RidgeRegressionModel_M" + to_string(m);
        M = (double)m;
        model.setting(m);
        model.load(m);
        model.predict(x3, 0);
        fn = "homework/Q4/curve_M" + to_string(m) + ".csv";
        x3.y_predict.save(fn);
    }
    return 0;
}