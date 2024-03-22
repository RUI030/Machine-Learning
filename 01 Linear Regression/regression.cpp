#include "regression.h"
#include <cmath>

using namespace std;

LinearRegression::LinearRegression(int m)
{
    M = m;
    s = 0.1;
    u.clear();
    u.push_back(0.0); // reduncdant term for convenience
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
    u.clear();
}
void LinearRegression::prep(dataset &ds, int i)
{
    if (i > ds.n)
        cout << "\033[1;31m< Model Preprocessing >[FAIL]: Index out of bounds.\033[0m" << endl;
    ds.split(train, valid, i);
    // save the parameters for normalization
    train.update();
    mean = train.x.Mean;
    sd = train.x.STD;
    valid.x.normalize(mean, sd);
    train.x.normalize();
    std::cout << "\033[1;32mSuccessfully preprocessed the data!\033[0m" << std::endl;
}
void LinearRegression::prep(dataset &ds)
{
    prep(ds, (int)(ds.n * 0.8));
}
void LinearRegression::update(const matrix &X, const matrix &t)
{
    int N = X.row(); // number of row of data
    int K = X.col(); // number of features
    // resize design matrix
    PHI.resize(N, K * M);
    // calculate design matrix
    for (int i = 0; i < N; i++) // number of rows
    {
        for (int k = 0; k < K; k++) // concatenate the design matrix of different feature
        {
            PHI[i][k * M] = 1; // phi(x) = 1 while j = 0 ==> reduncdant item for convenience
            for (int j = 1; j < M; j++)
            {
                PHI[i][k * M + j] = 1 / (1 + exp((X[i][k] - u[j]) / s));
            }
        }
    }
    // calculate wML
    matrix PHI_T;
    PHI_T = PHI;
    PHI_T.T();
    wML = PHI_T * PHI;
    wML.inv();
    wML.dot(PHI_T);
    wML.dot(t);
    std::cout << "\033[1;32mSuccessfully updated wML!\033[0m" << std::endl;
}
void LinearRegression::update(const dataset &ds)
{
    update(ds.x, ds.y);
}
void LinearRegression::update()
{
    update(train);
}
void LinearRegression::predict(dataset &ds)
{
    // only update y_predict
    ds.designMatrix(M, s, u);
    ds.y_predict = ds.PHI * wML;
    cout << "y_predicted: " << ds.y_predict.dim() << endl;
}
void LinearRegression::eval(dataset &ds)
{
    cout<<"...evaluating..."<<endl;
    predict(ds);
    int Nd = ds.y.row();
    int K = ds.y.col();
    ds.accuracy.resize(ds.y.col());
    double tmp;
    for (int i = 0; i < K; i++)
    {
        tmp = 0;
        for (int j = 0; j < Nd; j++)
        {
            tmp += abs(ds.y[i][j] - ds.y_predict[i][j]) / ds.y[i][j];
        }
        ds.accuracy[i] = 1 - tmp / Nd;
    }
    // print accuracy
    cout << "\n===================================\n";
    cout << "[ACCURACY]:";
    for (int i = 0; i < ds.accuracy.size(); i++)
    {
        cout << "\t" << ds.accuracy[i];
    }
    cout << "\n===================================\n";
}
void LinearRegression::eval()
{
    eval(valid);
}