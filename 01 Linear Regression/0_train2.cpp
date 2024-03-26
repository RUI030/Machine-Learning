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

using namespace std;

int main()
{
    dataset song, train, valid;
    LinearRegression model;
    string name = "RidgeRegressionModel";

    song.read("HW1.csv"); // load data
    model.split(song, 10000);
    model.prep();
    for (int m = 5; m <= 30; m += 5)
    {
        model.setting(m);
        model.rename(name);
        model.update2();
        model.save();
    }
    return 0;
}
