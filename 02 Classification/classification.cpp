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
void GenerativeModel::relabel(double source, double target)
{
    train.relabel(source, target);
    valid.relabel(source, target);
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
    k = 0;
    for (int i = 0; i < t.size(); i++)
    {
        if (t[i] > k)
            k = t[i];
    }
    k++;
    N = X.row();
    nf = X.col(); 
    Nk.clear();
    Nk.resize(k, 0); // k = 4
    for (int i = 0; i < N; i++)
    {
        Nk[t[i]]++;
    }
    for (int i = 0; i < k; i++)
    {
        pi.push_back((double)Nk[i] / (double)N);
    }
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
    // Calculate LAMBDA (inverse of SIGMA)
    MatrixXd lambda = sigma.inverse();
    // Convert Eigen::MatrixXd back to custom matrix class for LAMBDA
    LAMBDA.resize(nf, nf);
    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < nf; ++j) {
            LAMBDA[i][j] = lambda(i, j);
        }
    }
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
    matrix scores(ds.n, k);
    ds.y_predict.resize(ds.n, 1);
    for (int i = 0; i < ds.n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            matrix x_minus_mu_j(nf, 1);
            for (int m = 0; m < nf; m++)
            {
                // Subtract the class mean from each feature of the data point
                x_minus_mu_j[m][0] = ds.x[i][m] - mu[m][j];
            }

            // Transpose the x_minus_mu_j to multiply with LAMBDA
            matrix x_minus_mu_j_T = x_minus_mu_j; 
            x_minus_mu_j_T.T();

            // Calculate the score for class j
            matrix score_j = x_minus_mu_j_T * LAMBDA * x_minus_mu_j; 
            score_j.scale(-0.5); // Scale the score by -0.5
            scores[i][j] = score_j[0][0] + log(pi[j]); // Add the log prior probability
        }
    }

    // Find the class with the highest score for each data point
    for (int i = 0; i < ds.n; i++)
    {
        int maxpos = 0;
        for (int j = 1; j < k; j++)
        {
            if (scores[i][j] > scores[i][maxpos])
            {
                maxpos = j;
            }
        }
        ds.y_predict[i][0] = maxpos;
    }
}
void GenerativeModel::eval(dataset &ds)
{
    predict(ds);
    ds.k = k;
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

// Discriminative Classification Model
DiscriminativeModel::DiscriminativeModel(/* args */)
{
    name = "DiscriminativeModel";
    K = 4;
    M = 3;
    setting(0.01, 100, 20);
    w.resize(M, K, 0.0);
}
DiscriminativeModel::~DiscriminativeModel()
{
}
void DiscriminativeModel::relabel(double source, double target)
{
    train.relabel(source, target);
    valid.relabel(source, target);
}
void DiscriminativeModel::setting(double _lr, int _batch_size, int _epoch)
{
    lr = _lr;
    batch_size = _batch_size;
    epoch = _epoch;
}
void DiscriminativeModel::batchPredict(dataset &ds, int start, int end)
{
    if ((ds.y_k.col()!=K)||(ds.y_k.row()!=ds.n))
    {
        ds.y_k.resize(ds.n, K);
    }
    double sum;
    int n;
    vector<double> a; a.resize(K, 0.0);
    for (int _n = start; _n < end; _n++)
    {
        n = _n % ds.n;
        a.clear();
        sum = 0;
        for (int i = 0; i < K; i++)
        {
            a[i] = w[0][i];
            for (int j = 1; j < M; j++)
            {
                a[i] += w[j][i] * ds.x[n][j];
            }
            a[i] = exp(a[i]);
            sum += a[i];
        }
        for (int i = 0; i < K; i++)
        {
            ds.y_k[n][i] = a[i] / sum;
        }
    }
}
void DiscriminativeModel::batchUpdate(dataset &ds, int start, int end)
{
    batchPredict(ds, start, end);
    matrix grad; grad.resize(M, K, 0.0); // [M][K]
    vector<double> coef; coef.resize(K, 0.0); // [K]
    int n;
    for (int _n = start; _n < end; _n++)
    {
        n = _n % ds.n;
        for (int i = 0; i < K; i++)
        {
            coef[i] = ds.y_k[n][i];
        }
        coef[(int)ds.y[n][0]] -= 1;
        for (int i = 0; i < K; i++)
        {
            grad[0][i] += coef[i];
            for (int j = 1; j < M; j++)
            {
                grad[j][i] += coef[i] * ds.x[n][j-1];
            }
        }
    }
    // update w
    double scalar = 1.0 / (double)(end - start);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            w[i][j] -= lr * grad[i][j] * scalar;
        }
    }
}
void DiscriminativeModel::update()
{
    int iter = 0, batch_num;
    double best_acc = 0;
    batch_num = train.n / batch_size;
    while (iter < epoch)
    {
        for (int i = 0; i < batch_num; i++)
        {
            batchUpdate(train, i * batch_size, (i + 1) * batch_size);
            // if (acc>0.9) {save;break;}
            // if (acc>best_acc) save(_name);
        }
        batchUpdate(train, batch_num * batch_size, train.n);
        cout << "Epoch " << iter << ":";
        eval();
        iter++;
    }
}
void DiscriminativeModel::predict(dataset &ds)
{
    batchPredict(ds, 0, ds.n);
    ds.y_predict.resize(ds.n, 1);
    double maxpos; // This should be an integer, not a double.
    for (int i = 0; i < ds.n; i++)
    {
        maxpos = 0;
        for (int j = 1; j < K; j++)
        {
            if (ds.y_k[i][j] > ds.y_k[i][maxpos])
            {
                maxpos = j; // This should be inside the loop.
            }
        }
        ds.y_predict[i][0] = maxpos; // This needs to be inside the loop, right after finding maxpos.
    }
}
void DiscriminativeModel::eval(dataset &ds)
{
    predict(ds);
    ds.k = K;
    ds.ConfusionMatrix();
    cout << "\033[1;33mConfusion ";
    ds.confusion_matrix.print();
    int correct = 0;
    double acc;
    for (int i = 0; i < K; i++)
    {
        correct += ds.confusion_matrix[i][i];
    }
    acc = (double)correct / (double)ds.n;
    if (ds.accuracy.empty())
        ds.accuracy.push_back(acc);
    else
        ds.accuracy[0] = acc;
}
void DiscriminativeModel::eval()
{
    cout << " \033[36m>>> Train >>> \033[0m";
    eval(train);
    cout << train.accuracy[0];
    cout << " \033[35m<<< Valid <<< \033[0m";
    eval(valid);
    cout << valid.accuracy[0];
    cout << endl;
}