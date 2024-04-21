#include <iostream>
#include "classification.h"

int main()
{
    GenerativeModel GenModel;
    GenModel.train.read("HW2_training.csv");
    GenModel.valid.read("HW2_testing.csv");
    GenModel.update();
    GenModel.eval();
    return 0;
}