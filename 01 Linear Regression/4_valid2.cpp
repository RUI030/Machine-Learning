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

// consider transpose before saving the file????

int main()
{
    dataset hw1;
    LinearRegression model;
    double M, J;
    string fn, pre;
    string mName;
    matrix mse(0, 0), acc(0, 0) ;

    // load data
    hw1.read("HW1.csv");
    model.split(hw1, 10000);
    model.prep();

    // predict
    for (int m = 5; m <= 30; m += 5)
    {
        mName = "RidgeRegressionModel_M" + to_string(m);
        M = (double)m;
        model.setting(m);
        model.load(mName);
        model.eval();
        pre = "homework/Q4/" + model.name;
        model.train.y.save(pre + "_train_y.csv");
        model.train.y_predict.save(pre + "_train_y_predicted.csv");
        model.valid.y.save(pre + "_valid_y.csv");
        model.valid.y_predict.save(pre + "_valid_y_predicted.csv");
        mse.append(model.train.MSE);
        mse.append(model.valid.MSE);
        acc.append(model.train.accuracy);
        acc.append(model.valid.accuracy);
    }
    mse.save("homework/Q4/4_2_MSE.csv");
    acc.save("homework/Q4/4_3_Accuracy.csv");
    return 0;
}