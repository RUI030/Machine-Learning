#include "regression.h"
#include <cmath>
#include <string>
#include <iomanip>

using namespace std;

LinearRegression::LinearRegression(int m)
{
    setting(m);
}
LinearRegression::LinearRegression()
{
    LinearRegression(5);
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
    mean_y = train.y.Mean;
    sd_y = train.y.STD;
    valid.x.normalize(mean, sd);
    train.x.normalize();
    // valid.y.normalize(mean_y, sd_y);
    // train.y.normalize();
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
void LinearRegression::update2(matrix &X, const matrix &t)
{
    int N = X.row(); // Number of rows of data
    int K = X.col(); // Number of features

    // Calculate design matrix
    PHI.resize(N, K * M);
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < K; k++)
        {
            PHI[i][k * M] = 1; // phi(x) = 1 for j = 0 (redundant term for convenience)
            for (int j = 1; j < M; j++)
            {
                PHI[i][k * M + j] = 1 / (1 + exp(-(X[i][k] - u[j]) / s)); // basis function
            }
        }
    }

    // Ridge regression using SVD
    matrix U, Sigma, V;
    PHI.svd(U, Sigma, V);

    // Calculate the pseudo-inverse of Sigma with ridge regularization
    matrix Sigma_inv(Sigma.col(), Sigma.row());
    for (int i = 0; i < min(Sigma.row(), Sigma.col()); i++)
    {
        Sigma_inv[i][i] = Sigma[i][i] / (Sigma[i][i] * Sigma[i][i] + lambda);
    }

    // Calculate wML = V * Sigma_inv * U^T * t
    U.T();
    U.dot(t);
    wML = V * Sigma_inv * U;

    std::cout << "\033[1;32mSuccessfully updated wML with ridge regression!\033[0m" << std::endl;
}
void LinearRegression::update2(dataset &ds)
{
    update2(ds.x, ds.y);
}
void LinearRegression::update2()
{
    update2(train);
}
double LinearRegression::basisFunction(double val, int k, int j, int m, double S, double uj)
{
    return (j) ? (1.0 / (1.0 + exp(-(val - uj) / s))) : 1;
}
double LinearRegression::basisFunction(double val, int k, int j, int m)
{
    return basisFunction(val, k, j, m, s, u[j]);
}
double LinearRegression::basisFunction(double val, int k, int j)
{
    return basisFunction(val, k, j, M, s, u[j]);
}
void LinearRegression::predict(dataset &ds)
{
    // std::cout << "...predicting..." << std::endl;
    // only update y_predict
    ds.designMatrix(M, s, u);
    // cout << "PHI_val_dim:" << ds.PHI.dim() << "\twML_dim:" << wML.dim() << endl;
    ds.y_predict = ds.PHI * wML;
    // std::cout << "y_predicted: " << ds.y_predict.dim() << std::endl;
    // std::cout << "\033[1;32mSuccessfully predicted the output!\033[0m" << std::endl;
}
void LinearRegression::eval(dataset &ds, bool doNorm)
{
    if (doNorm)
        normalize(ds);
    predict(ds);
    // std::cout << "...evaluating..." << std::endl;
    int Nd = ds.y.row();
    int K = ds.y.col();
    // std::cout << "Nd: " << Nd << "\tK: " << K << std::endl;
    double tmp, ttmp, tmp1, ttmp1, nd, a, b;
    nd = (double)Nd;
    ds.accuracy.resize(K);
    ds.MSE.resize(K);
    // cout << "Y_dim:" << ds.y.dim() << "\tY_pred_dim:" << ds.y_predict.dim() << endl;
    // for (int i = 0; i < 10; i++)
    //     std::cout << "Y: " << ds.y[i][0] << "\tY_pred: " << ds.y_predict[i][0] << endl;
    for (int j = 0; j < K; j++)
    {
        tmp = 0.0;
        tmp1 = 0.0;
        for (int i = 0; i < Nd; i++)
        {
            // if y too small it cause inf!!!
            a = ds.y[i][j];
            b = ds.y_predict[i][j];
            ttmp = a ? abs((a - b) / a) : abs(a - b);
            ttmp1 = pow((a - b), 2);
            // tmp += (ttmp > 1.0) ? 1.0 : ttmp;
            tmp += ttmp;
            tmp1 += ttmp1;
            // print large error
            // if (ttmp > 1.0)
            //     cout << a << "\t" << b << "\t" << ttmp << endl;
        }
        // cout << "tmp: " << tmp << endl;
        tmp /= nd;
        tmp1 /= nd;
        ds.accuracy[j] = 1.0 - tmp;
        ds.MSE[j] = tmp1;
    }

    // Print accuracy
    // std::cout << "\n===================================\n";
    std::cout << name << "\t[ACC]:";
    for (size_t i = 0; i < ds.accuracy.size(); i++)
    {
        std::cout << setw(12) << ds.accuracy[i];
    }
    std::cout << "\t[MSE]:";
    for (size_t i = 0; i < ds.accuracy.size(); i++)
    {
        std::cout << setw(12) << ds.MSE[i];
    }
    // std::cout << "\n===================================\n";
}
void LinearRegression::eval(dataset &ds)
{
    eval(ds, 1);
}
void LinearRegression::eval()
{
    // cout << "\n\033[36m>>>>>>>>>>>>>>>>>>> Train >>>>>>>>>>>>>>>>>>>\033[0m\n";
    cout << "\n\033[36m>>> Train >>> \033[0m";
    eval(train, 0);
    // cout << "\n\033[35m<<<<<<<<<<<<<<<<<<< Valid <<<<<<<<<<<<<<<<<<<\033[0m\n";
    cout << "\n\033[35m<<< Valid <<< \033[0m";
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
void LinearRegression::setting(const int _m, const double _s, const double l)
{
    M = (double)_m;
    s = _s;
    lambda = l;
    name = "RegressionModel_M" + to_string(_m);
    u.clear();
    u.push_back(0.0); // reduncdant term for convenience
    for (double j = 1; j < M; j++)
    {
        u.push_back((3.0 * (-M + 1 + 2 * (j - 1) * (M - 1) / (M - 2))) / M);
    }
}
void LinearRegression::setting(const int _m, const double _s)
{
    setting(_m, 0.1, 0.1);
}
void LinearRegression::setting(const int _m)
{
    setting(_m, 0.1, 0.1);
}
void LinearRegression::rename(const std::string &modelName)
{
    this->name = modelName + "_M" + to_string((int)M);
}
void LinearRegression::load(const std::string &modelName, const std::string &pre)
{
    std::string tmpName = modelName;
    std::string _wML = pre + modelName + ".csv";
    std::string _set = pre + modelName + "_setting.csv";
    std::string _mean = pre + modelName + "_mean.csv";
    std::string _sd = pre + modelName + "_sd.csv";
    // weights
    wML.read(_wML);
    // setting: gaussian noise (basis function) parameter
    matrix tmp;
    tmp.read(_set);
    M = tmp[0][0];
    s = tmp[0][1];
    lambda = tmp[0][2];
    setting(M, s);
    name = tmpName;
    // mean ans std
    matrix MEAN, SD;
    MEAN.read(_mean);
    SD.read(_sd);
    mean = MEAN.data[0];
    sd = SD.data[0];
}
void LinearRegression::load(const std::string &modelName)
{
    load(modelName, "model/");
}
void LinearRegression::load(const int _m)
{
    load(this->name, "model/");
}
void LinearRegression::save(const std::string &modelName, const std::string &pre)
{
    std::string _wML = pre + modelName + ".csv";
    std::string _set = pre + modelName + "_setting.csv";
    std::string _mean = pre + modelName + "_mean.csv";
    std::string _sd = pre + modelName + "_sd.csv";
    matrix tmp(1, 3);
    tmp[0][0] = M;
    tmp[0][1] = s;
    tmp[0][2] = lambda;
    matrix MEAN, SD;
    MEAN.append(mean);
    SD.append(sd);
    tmp.save(_set);
    wML.save(_wML);
    MEAN.save(_mean);
    SD.save(_sd);
}
void LinearRegression::save(const std::string &modelName)
{
    save(modelName, "model/");
}
void LinearRegression::save()
{
    save(name);
}