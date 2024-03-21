#include "regression.h"
#include <cmath>

LinearRegression::LinearRegression(int m)
{
    M = m;
    s = 0.1;
    u.clear();
    for (int j = 1; j < M; j++)
    {
        u.push_back((-M + 1 + 2 * (j - 1) * (M - 1) / (M - 2)) * 3 / M);
    }
}
LinearRegression::LinearRegression()
{
    LinearRegression(15);
}
LinearRegression::~LinearRegression()
{
}
double LinearRegression::basisFunction(const int k, const int j, const int xk)
{
    return j ? 1.0 : 1/(1+exp((u[j]-xk)/s));
}