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
#include "regression.h"
#include "matrix.h"

using namespace std;

int main()
{
    dataset song;
    LinearRegression model;
    song.read("HW1.csv"); // load data
    model.prep(song, 10000);
    matrix mse(0, 0), acc(0, 0) ;
    for (int m = 5; m <= 30; m += 5)
    {
        model.setting(m);
        model.load(m);
        model.eval();
        mse.append(model.train.MSE);
        mse.append(model.valid.MSE);
        acc.append(model.train.accuracy);
        acc.append(model.valid.accuracy);
        // todo: save evaluated result:
        // hmmm... need to change regression.h and .cpp
    }
    mse.save("homework/Q2/2_1_MSE.csv");
    acc.save("homework/Q2/2_2_ACC.csv");
    return 0;
}
