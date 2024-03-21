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
#include "regression.h"

using namespace std;

int main()
{
    dataset song, train, valid;
    song.read("HW1.csv");
    // song.print();
    song.split(train,valid,10000);
    // Preprocessing
    train.x.normalize();
    valid.x.normalize(train.x);
    train.x.update();
    valid.x.update();
    
    return 0;
}
