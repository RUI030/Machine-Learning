#include "regression.h"
#include <cmath>

using namespace std;

LinearRegression::LinearRegression(int m)
{
    M = m;
    s = 0.1;
    u.clear();
    u.push_back(0.0); // reduncdant term for convenience
    double Md = (double)m;
    for (double j = 1; j < M; j++)
    {
        u.push_back((3.0 * (-Md + 1 + 2 * (j - 1) * (Md - 1) / (Md - 2))) / Md);
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
// void LinearRegression::update(const matrix &X, const matrix &t)
// {
//     int N = X.row(); // number of rows of data
//     int K = X.col(); // number of features
//     // resize design matrix
//     PHI.resize(N, K * M);
//     // calculate design matrix
//     for (int i = 0; i < N; i++) // number of rows
//     {
//         for (int k = 0; k < K; k++) // concatenate the design matrix of different feature
//         {
//             PHI[i][k * M] = 1; // phi(x) = 1 while j = 0 ==> redundant item for convenience
//             for (int j = 1; j < M; j++)
//             {
//                 PHI[i][k * M + j] = 1 / (1 + exp((X[i][k] - u[j]) / s));
//             }
//         }
//     }
//     // calculate wML using SVD
//     matrix U, Sigma, V;
//     PHI.svd(U, Sigma, V);
//     // Calculate the pseudo-inverse of Sigma
//     matrix Sigma_inv(Sigma.col(), Sigma.row());
//     for (int i = 0; i < min(Sigma.row(), Sigma.col()); i++)
//     {
//         if (Sigma[i][i] > 1e-6) // threshold for singular values
//         {
//             Sigma_inv[i][i] = 1.0 / Sigma[i][i];
//         }
//         else
//         {
//             Sigma_inv[i][i] = 0.0;
//         }
//     }
//     // Calculate wML = V * Sigma_inv * U^T * t
//     wML = V * Sigma_inv;
//     U.T();
//     wML.dot(U);
//     wML.dot(t);
//     std::cout << "\033[1;32mSuccessfully updated wML!\033[0m" << std::endl;
// }
void LinearRegression::update(const matrix &X, const matrix &t)
{
    int N = X.row(); // Number of rows of data
    int K = X.col(); // Number of features

    // Resize design matrix
    PHI.resize(N, K * M);

    // Calculate design matrix
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < K; k++)
        {
            PHI[i][k * M] = 1; // phi(x) = 1 for j = 0 (redundant term for convenience)
            for (int j = 1; j < M; j++)
            {
                PHI[i][k * M + j] = 1 / (1 + exp(-(X[i][k] - u[j]) / s)); // note the negative sign in the exponent
            }
        }
    }

    // Calculate wML using SVD
    matrix U, Sigma, V;
    PHI.svd(U, Sigma, V);

    // Calculate the pseudo-inverse of Sigma
    matrix Sigma_inv(Sigma.col(), Sigma.row());
    for (int i = 0; i < min(Sigma.row(), Sigma.col()); i++)
    {
        Sigma_inv[i][i] = (Sigma[i][i] > 1e-4) ? 1.0 / Sigma[i][i] : 0.0;
    }

    // Calculate wML = V * Sigma_inv * U^T * t
    U.T();
    U.dot(t);
    wML = V * Sigma_inv * U;

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
    std::cout << "...predicting..." << std::endl;
    // only update y_predict
    ds.designMatrix(M, s, u);
    cout << "PHI_val_dim:" << ds.PHI.dim() << "\twML_dim:" << wML.dim() << endl;
    ds.y_predict = ds.PHI * wML;
    std::cout << "y_predicted: " << ds.y_predict.dim() << std::endl;
    std::cout << "\033[1;32mSuccessfully predicted the output!\033[0m" << std::endl;
}
void LinearRegression::eval(dataset &ds, bool doNorm)
{
    if (doNorm)
        normalize(ds);
    predict(ds);
    std::cout << "...evaluating..." << std::endl;
    int Nd = ds.y.row();
    int K = ds.y.col();
    std::cout << "Nd: " << Nd << "\tK: " << K << std::endl;
    double tmp, ttmp, nd, a, b;
    nd = (double)Nd;
    ds.accuracy.resize(K);
    cout << "Y_dim:" << ds.y.dim() << "\tY_pred_dim:" << ds.y_predict.dim() << endl;
    for (int i = 0; i < 10; i++)
        std::cout << "Y: " << ds.y[i][0] << "\tY_pred: " << ds.y_predict[i][0] << endl;
    for (int j = 0; j < K; j++)
    {
        tmp = 0.0;
        for (int i = 0; i < Nd; i++)
        {
            // if y too small it cause inf!!!
            a = ds.y[i][j];
            b = ds.y_predict[i][j];
            ttmp = a ? abs((a - b) / a) : abs(a - b);
            // tmp += (ttmp > 1.0) ? 1.0 : ttmp;
            tmp += ttmp;
            if (ttmp > 1.0)
                cout << a << "\t" << b << "\t" << ttmp << endl;
        }
        cout << "tmp: " << tmp << endl;
        tmp /= nd;
        ds.accuracy[j] = 1.0 - tmp;
    }

    // Print accuracy
    std::cout << "\n===================================\n";
    std::cout << "[ACCURACY]:";
    for (size_t i = 0; i < ds.accuracy.size(); i++)
    {
        std::cout << "\t" << ds.accuracy[i];
    }
    std::cout << "\n===================================\n";
}
void LinearRegression::eval(dataset &ds)
{
    eval(ds, 1);
}
void LinearRegression::eval()
{
    cout << "\nTrain ======================================\n";
    eval(train, 0);
    cout << "\nValid ======================================\n";
    eval(valid, 0);
}
void LinearRegression::normalize(matrix &input)
{
    input.normalize(mean, sd);
}
void LinearRegression::normalize(dataset &input)
{
    normalize(input.x);
}