#include <iostream>
#include "classification.h"

int main()
{

    DiscriminativeModel model_D;
    model_D.train.read("HW2_training.csv");
    model_D.valid.read("HW2_testing.csv");
    // model_D.setting(0.001, 200, 1000);     // 0.5  0.5 
    // model_D.setting(0.0001, 500, 1000);    // 0.65 0.59
    model_D.setting(0.0001, 500, 1000);
    model_D.update();

    // GenerativeModel model_G;
    // model_G.train.read("HW2_training.csv");
    // model_G.valid.read("HW2_testing.csv");
    // model_G.update();
    // model_G.eval();
    return 0;
}