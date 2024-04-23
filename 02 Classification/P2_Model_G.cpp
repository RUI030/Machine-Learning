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
    model_G.train.relabel(3, 0);
    model_G.valid.relabel(3, 0);
    // Part 2.
    model_G.rename("GenerativeModel2");
    model_G.k = 3;
    model_G.update();
    model_G.eval();
    model_G.save();
    model_G.predict(scatter);
    scatter.y_predict.save("homework/p2_g_scatter_y.csv");
    model_G.train.confusion_matrix.save("homework/p2_g_train_cm.csv");
    model_G.valid.confusion_matrix.save("homework/p2_g_valid_cm.csv");
    return 0;
}