#include <iostream>
#include "classification.h"

int main()
{
    GenerativeModel model_G;
    dataset scatter; matrix X;
    X.read("homework/scatter_x.csv");
    scatter.x = X; scatter.n = scatter.x.row();
    // Read data
    model_G.train.read("HW2_training.csv");
    model_G.valid.read("HW2_testing.csv");
    // normalize x
    // model_G.train.x.normalize();
    // model_G.valid.x.normalize();
    // scatter.x.normalize();
    // Part 1.
    model_G.update();
    model_G.eval();
    model_G.save();
    model_G.predict(scatter);
    scatter.y_predict.save("homework/p1_g_scatter_y.csv");
    model_G.train.confusion_matrix.save("homework/p1_g_train_cm.csv");
    model_G.valid.confusion_matrix.save("homework/p1_g_valid_cm.csv");
    return 0;
}