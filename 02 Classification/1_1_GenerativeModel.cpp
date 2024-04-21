#include <iostream>
#include "classification.h"

int main()
{
    GenerativeModel model_G;
    model_G.train.read("HW2_training.csv");
    model_G.valid.read("HW2_testing.csv");
    model_G.update();
    model_G.eval();
    return 0;
}