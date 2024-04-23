#include <iostream>
#include "classification.h"

int main()
{
    cout << "\n\033[1;37m###### Generative Model ######\033[0m\n" << endl;
    GenerativeModel model_G;
    model_G.train.read("HW2_demo_training.csv");
    model_G.valid.read("HW2_demo_testing.csv");
    model_G.update();
    model_G.eval();

    cout << "\n\033[1;37m###### Discriminative Model######\033[0m\n" << endl;
    DiscriminativeModel model_D;
    model_D.train.read("HW2_demo_training.csv");
    model_D.valid.read("HW2_demo_testing.csv");
    model_D.setting(0.001, 500, 3000);
    model_D.update(0.85, 100);
    model_D.eval(model_D.train);
    model_D.eval(model_D.valid);

    return 0;
}