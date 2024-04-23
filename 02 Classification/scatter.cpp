#include <iostream>
#include "classification.h"

int main()
{
    dataset scatter; matrix X;
    int count = 100;
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < count; j++)
        {
            vector<double> x(2);
            x[0] = i; x[1] = j;
            X.append(x);
        }
    }
    scatter.x = X; scatter.n = count * count; scatter.xdim = 2;
    scatter.x.save("homework/scatter_x.csv");

    return 0;
}