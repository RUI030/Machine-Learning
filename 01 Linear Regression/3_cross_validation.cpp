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
    int k = 5;
    dataset hw1, sub[5];
    LinearRegression model;
    hw1.read("HW1.csv"); // load data
    // split into K subset
    int cut[6];    // index to split the subset
    int n = 10000; // first 1000 for training and cross validation
    cut[0] = 0;
    cut[5] = n;
    for (int c = 1; c <= k; c++)
    {
        cut[c] = c * n / k;
        sub[c - 1].copy(hw1, cut[c - 1], cut[c]);
    }
    // K fold
    for (int c = 0; c < k; c++)
    {
        // clear
        // put the data in
        // preprocessing
        // train
        for (int m = 5; m <= 30; m += 5)
        {
            model.setting(m);
            model.update();
        }
        // record Loss Function, which could calculate from MSE since the given error function = 1/2 SUM((y-t)^2) = Nd/2 * MSE
    }
    // save every Loss 
    // Choose best Model
    // Train & save model
    
    return 0;
}
