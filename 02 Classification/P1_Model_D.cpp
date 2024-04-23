#include <iostream>
#include "classification.h"

int main()
{
    DiscriminativeModel model_D;
    dataset scatter; matrix X;
    X.read("homework/scatter_x.csv");
    scatter.x = X; scatter.n = scatter.x.row();
    // Read data
    model_D.train.read("HW2_training.csv");
    model_D.valid.read("HW2_testing.csv");
    // Part 1.
    model_D.setting(0.001, 500, 3000);
    model_D.update(0.85, 100);
    model_D.eval();
    model_D.predict(scatter);
    scatter.y_predict.save("homework/p1_d_scatter_y.csv");
    model_D.train.confusion_matrix.save("homework/p1_d_train_cm.csv");
    model_D.valid.confusion_matrix.save("homework/p1_d_valid_cm.csv");
    model_D.saveLog("p1_d");
    return 0;
}