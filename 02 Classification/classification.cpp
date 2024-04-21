#include "classification.h"
#include "matrix.h"
#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <Eigen/Dense>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Generative Classification Model
GenerativeModel::GenerativeModel(/* args */)
{
    name = "GenerativeModel";
}
GenerativeModel::~GenerativeModel()
{
}
void GenerativeModel::relabel(dataset &ds, int source, int target)
{
    for (int i = 0; i < ds.n; i++)
    {
        if (ds.y[i][0] == source)
            ds.y[i][0] = target;
    }
    
}
void GenerativeModel::relabel(int source, int target)
{
    relabel(train, source, target);
    relabel(valid, source, target);
}
void GenerativeModel::update(matrix &X, matrix &_t)
{
    vector<int> t;
    t.clear();
    for (int i = 0; i < _t.row(); i++)
    {
        t.push_back((int)_t[i][0]);
    }
    // Calculate the number of classes
    // k = 0;
    // for (int i = 0; i < t.size(); i++)
    // {
    //     if (t[i] > k)
    //         k = t[i];
    // }
    // k++;
    k = 4;
    N = X.row(); // Number of rows of data
    nf = X.col(); // Number of features
    cout << "Number of data: " << N << endl;
    cout << "Number of features: " << nf << endl;
    // number of data for each class
    Nk.clear();
    Nk.resize(4, 0); // k = 4
    for (int i = 0; i < N; i++)
    {
        Nk[t[i]]++;
    }
    cout << "Update Nk success"<<endl;
    // pi (the prior probability of each class)
    for (int i = 0; i < k; i++)
    {
        pi.push_back((double)Nk[i] / (double)N);
    }
    cout << "Update pi success"<<endl;
    // mu (mean of each class of each feature)
    mu.resize(nf, k);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < nf; j++)
        {
            mu[j][t[i]] += X[i][j];
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < nf; j++)
        {
            mu[j][i] /= Nk[i];
        }
    }
    cout << "Update mu success"<<endl;
    // Calculate SIGMA
    MatrixXd sigma = MatrixXd::Zero(nf, nf); // Initialize sigma matrix in Eigen
    vector<MatrixXd> classSigma(k, MatrixXd::Zero(nf, nf)); // Per-class sigma matrices
    for (int i = 0; i < N; ++i) {
        int classIdx = t[i];
        VectorXd x_i = VectorXd::Zero(nf);
        VectorXd mu_k = VectorXd::Zero(nf);

        for (int j = 0; j < nf; ++j) {
            x_i(j) = X[i][j];
            mu_k(j) = mu[j][classIdx];
        }

        VectorXd x_mu = x_i - mu_k;
        classSigma[classIdx] += x_mu * x_mu.transpose();
    }
    for (int i = 0; i < k; ++i) {
        if (Nk[i] > 1) {
            classSigma[i] /= (Nk[i] - 1);
        }
        sigma += classSigma[i] * pi[i];
    }
    cout << "Update EIGEN success"<<endl;
    // Convert Eigen::MatrixXd back to custom matrix class
    SIGMA.resize(nf, nf);
    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < nf; ++j) {
            SIGMA[i][j] = sigma(i, j);
        }
    }
    // Calculate LAMBDA (inverse of SIGMA)
    MatrixXd lambda = sigma.inverse();
    // Convert Eigen::MatrixXd back to custom matrix class for LAMBDA
    LAMBDA.resize(nf, nf);
    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < nf; ++j) {
            LAMBDA[i][j] = lambda(i, j);
        }
    }
    cout << "Update LAMBDA success"<<endl;
    // weight = LAMBDA * mu
    w = LAMBDA * mu;
    // bias = [w_{k0}] = -0.5 * mu_k^T * SIGMA * mu_k + log(pi_k)
    w0.resize(1, k);
    matrix mu_k, mu_k_T;
    for (int i = 0; i < k; i++)
    {
        mu_k.slice(mu, 0 , mu.row(), i,i+1);
        mu_k_T = mu_k; mu_k_T.T();
        mu_k_T.dot(LAMBDA);
        mu_k_T.dot(mu_k);
        w0[0][i] = -0.5 * mu_k_T[0][0] + log(pi[i]);
    }
    cout << "Update WEIGHTS success"<<endl;
}
void GenerativeModel::update(dataset &ds)
{
    update(ds.x, ds.y);
}
void GenerativeModel::update()
{
    update(train);
}
void GenerativeModel::predict(dataset &ds)
{
    matrix a, posterior(ds.n, k);
    a = ds.x * w;
    for (int i = 0; i < ds.n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            posterior[i][j] = sigmoid(a[i][j]+w0[0][j]);
        }
    }
    ds.y_predict.resize(ds.n, 1);
    for (int i = 0; i < ds.n; i++)
    {
        int maxpos = 0;
        for (int j = 0; j < k; j++)
        {
            if (posterior[i][j] > posterior[i][maxpos])
                ds.y_predict[i][0] = (double)j;
        }
    }
}
void GenerativeModel::eval(dataset &ds)
{
    predict(ds);
    ds.ConfusionMatrix();
    int correct = 0;
    double acc;
    for (int i = 0; i < k; i++)
    {
        correct += ds.confusion_matrix[i][i];
    }
    acc = (double)correct / (double)ds.n;
    if (ds.accuracy.empty())
        ds.accuracy.push_back(acc);
    else
        ds.accuracy[0] = acc;
    cout << "\033[1;37mAccuracy:" << acc << "\033[0m" << endl;
    cout << "\033[1;33mConfusion ";
    ds.confusion_matrix.print();
}
void GenerativeModel::eval()
{
    cout << "\n\033[36m>>>>>>>>>>>>>>>>>>> Train >>>>>>>>>>>>>>>>>>>\033[0m\n";
    eval(train);
    cout << endl;
    cout << "\n\033[35m<<<<<<<<<<<<<<<<<<< Valid <<<<<<<<<<<<<<<<<<<\033[0m\n";
    eval(valid);
    cout << endl;
}
void GenerativeModel::rename(const std::string &modelName)
{
    name = modelName;
}
void GenerativeModel::load(const std::string &modelName, const std::string &pre)
{
    std::string _w = pre + modelName + ".csv";
    matrix w_all;
    w_all.read(_w);
    w0.slice(w_all, 0, 1, 0, k);
    w.slice(w_all, 1, nf+1, 0, k);
}
void GenerativeModel::load(const std::string &modelName)
{
    load(modelName, "model/");
}
void GenerativeModel::save(const std::string &modelName, const std::string &pre)
{
    std::string _w = pre + modelName + ".csv";
    matrix w_all(nf+1, k);
    w_all.append(w0);
    w_all.append(w);
    w_all.save(_w);
}
void GenerativeModel::save(const std::string &modelName)
{
    save(modelName, "model/");
}
void GenerativeModel::save()
{
    save(name);
}