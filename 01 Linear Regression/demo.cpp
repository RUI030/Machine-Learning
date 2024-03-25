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
    int M = 5;
    dataset song;
    LinearRegression model(M);
    song.read("HW1_demo.csv"); // load data
    model.load(M);
    model.eval(song);
    return 0;
}
