#include <iostream>
#include "classification.h"

int main()
{
    dataset scatter; matrix X;
    int count = 1000;
    double div = 10;
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < count; j++)
        {
            vector<double> x(2);
            x[0] = (double)i/div; x[1] = (double)j/div;
            X.append(x);
        }
    }
    scatter.x = X; scatter.n = count * count; scatter.xdim = 2;
    scatter.x.save("homework/scatter_x.csv");

    return 0;
}