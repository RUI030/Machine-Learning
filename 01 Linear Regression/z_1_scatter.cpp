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
    matrix x3;
    double M, J;
    string fn, pre;

    // load data
    hw1.read("HW1.csv");
    // x3
    x3.slice(hw1.x, 0, hw1.x.row() , 2, 3);
    x3.save("homework/x3.csv");
    model.split(hw1, 10000);
    model.prep();

    // predict
    for (int m = 5; m <= 30; m += 5)
    {
        M = (double)m;
        model.setting(m);
        model.load(m);
        model.eval();
        pre = "homework/Q1/" + model.name;
        model.train.y.save(pre + "_train_y.csv");
        model.train.y_predict.save(pre + "_train_y_predicted.csv");
        model.valid.y.save(pre + "_valid_y.csv");
        model.valid.y_predict.save(pre + "_valid_y_predicted.csv");
    }
    return 0;
}