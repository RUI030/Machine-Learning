#include <iostream>
#include "classification.h"

int main()
{
    // DiscriminativeModel model_D;
    // dataset scatter; matrix X;
    // X.read("homework/scatter_x.csv");
    // scatter.x = X; scatter.n = scatter.x.row();
    // // Read data
    // model_D.train.read("HW2_training.csv");
    // model_D.valid.read("HW2_testing.csv");
    // // normalize x 
    // model_D.train.x.normalize();
    // model_D.valid.x.normalize();
    // scatter.x.normalize();
    // // Part 1.
    // model_D.setting(0.001, 200, 3000);
    // model_D.update(0.8, 100);
    // model_D.eval();
    // model_D.predict(scatter);
    // scatter.y_predict.save("tmp/p1_d_scatter_y.csv");
    // model_D.train.confusion_matrix.save("tmp/p1_d_train_cm.csv");
    // model_D.valid.confusion_matrix.save("tmp/p1_d_valid_cm.csv");
    // model_D.saveLog("p1_d");

    DiscriminativeModel model_D;
    dataset scatter; matrix X;
    X.read("homework/scatter_x.csv");
    scatter.x = X; scatter.n = scatter.x.row();
    // Read data
    model_D.train.read("HW2_training.csv");
    model_D.valid.read("HW2_testing.csv");
    model_D.train.relabel(3, 0);
    model_D.valid.relabel(3, 0);
    // normalize x 
    model_D.train.x.normalize();
    model_D.valid.x.normalize();
    scatter.x.normalize();
    // Part 2.
    model_D.rename("DiscriminativeModel2");
    model_D.K = 3;
    model_D.setting(0.0005, 200, 5000);
    model_D.update(0.9, 100);
    model_D.eval();
    model_D.predict(scatter);
    scatter.y_predict.save("tmp/p2_d_scatter_y.csv");
    model_D.train.confusion_matrix.save("tmp/p2_d_train_cm.csv");
    model_D.valid.confusion_matrix.save("tmp/p2_d_valid_cm.csv");
    model_D.saveLog("p2_d");
    return 0;
}