/*
===========================================

Compile:
g++ -std=c++11 -o main main.cpp regression.cpp dataset.cpp matrix.cpp

Execute:
./main

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
    LinearRegression model(5);
    song.read("HW1.csv"); // load data
    model.prep(song,10000);
    model.update();
    model.eval();
    return 0;
}
