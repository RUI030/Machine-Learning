#include <iostream>
#include "classification.h"

int main()
{

    DiscriminativeModel model_D;
    model_D.train.read("HW2_training.csv");
    model_D.valid.read("HW2_testing.csv");
    model_D.setting(0.01, 100, 10);
    model_D.update();

    // GenerativeModel model_G;
    // model_G.train.read("HW2_training.csv");
    // model_G.valid.read("HW2_testing.csv");
    // model_G.update();
    // model_G.eval();
    return 0;
}