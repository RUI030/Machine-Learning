
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
    string fn;
    vector<double> u;
    double min = 0.0, max = 1.0, tmp;

    // // ================================
    //    Find normalized, x_3 range
    // // ================================
    // dataset hw1;
    // matrix x;
    // hw1.read("HW1.csv");
    // hw1.normalize();
    // x = hw1.x;
    // double now;
    // for(int i=0;i<x.row();i++)
    // {
    //     now = x[i][2];
    //     if(now>max) max = now;
    //     if(now<min) min = now;
    // }
    // cout<<"[MAX]: "<<max<<endl;
    // cout<<"[MIN]: "<<min<<endl;

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
        M = (double)m;
        model.setting(m);
        model.load(m);
        model.predict(x3, 0);
        fn = "homework/Q1/curve_M" + to_string(m) + ".csv";
        x3.y_predict.save(fn);
    }
    return 0;
}