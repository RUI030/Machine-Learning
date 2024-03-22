#include <iostream>
#include <string>
#include <vector>
#include "regression.h"

using namespace std;

int main()
{
    dataset song;
    LinearRegression model(5);
    song.read("HW1.csv");           // load data
    model.prep(song,10000);
    model.update();
    return 0;
}
