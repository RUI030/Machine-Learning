#include "classification.h"
#include "matrix.h"
#include "dataset.h"
#include <iostream>
#include <iomanip>
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
    std::string _set = pre + modelName + "_setting.csv";
    std::string _pi = pre + modelName + "_pi.csv";
    std::string _mu = pre + modelName + "_mu.csv";
    std::string _L = pre + modelName + "_LAMBDA.csv";
    matrix w_all;
    w_all.read(_w);
    w0.slice(w_all, 0, 1, 0, k);
    w.slice(w_all, 1, nf+1, 0, k);
    matrix tmp;
    tmp.read(_set);
    k = (int)tmp[0][0];
    nf = (int)tmp[0][1];
    matrix PI;
    PI.read(_pi);
    pi = PI[0];
    mu.read(_mu);
    LAMBDA.read(_L);    
}
void GenerativeModel::load(const std::string &modelName)
{
    load(modelName, "model/");
}
void GenerativeModel::save(const std::string &modelName, const std::string &pre)
{
    std::string _w = pre + modelName + ".csv";
    std::string _set = pre + modelName + "_setting.csv";
    std::string _pi = pre + modelName + "_pi.csv";
    std::string _mu = pre + modelName + "_mu.csv";
    std::string _L = pre + modelName + "_LAMBDA.csv";
    // weights
    matrix w_all(nf+1, k);
    w_all.append(w0);
    w_all.append(w);
    w_all.save(_w);
    // k, nf
    matrix tmp(1, 2);
    tmp[0][0] = k;
    tmp[0][1] = nf;
    tmp.save(_set);
    // pi: prior probability
    matrix PI;
    PI.append(pi);
    PI.save(_pi);
    // mu
    mu.save(_mu);
    // LAMBDA
    LAMBDA.save(_L);
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
    setting(0.1, 100, 20);
    w.resize(M, K, 0.5);
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
    string _name = name + "_lr" + to_string(lr) + "_bs" + to_string(batch_size);
    rename(_name);
}
void DiscriminativeModel::batchPredict(dataset &ds, int start, int end)
{
    if ((ds.y_k.col()!=K)||(ds.y_k.row()!=ds.n))
    {
        ds.y_k.resize(ds.n, K);
    }
    double sum, tmp;
    int n;
    vector<double> a(K, 0.0); // Initialized directly
    for (int _n = start; _n < end; _n++)
    {
        n = _n % ds.n;
        sum = 0;
        for (int i = 0; i < K; i++)
        {
            tmp = w[0][i];
            for (int j = 1; j < M; j++)
            {
                tmp += w[j][i] * ds.x[n][j-1];
            }
            a[i] = exp(tmp);
            sum += a[i];
        }
        sum = sum == 0 ? 1e-10 : sum; // Avoid division by zero
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
    // cout << "Weights Updated: " << endl;
    // w.print();
}
void DiscriminativeModel::update()
{
    update_log.clear();
    int iter = 0, batch_num;
    double best_acc = 0.5;
    batch_num = train.n / batch_size;
    while (iter < epoch)
    {
        vector<double> _log;_log.clear();
        for (int i = 0; i < batch_num; i++)
        {
            batchUpdate(train, i * batch_size, (i + 1) * batch_size);
        }
        batchUpdate(train, batch_num * batch_size, train.n);
        cout << "Epoch " << iter << ":";
        eval();
        _log.push_back(train.accuracy[0]);
        _log.push_back(valid.accuracy[0]);
        update_log.append(_log);
        if (train.accuracy[0] > best_acc)
        {
            best_acc = train.accuracy[0];
            save(name + "_epoch" + to_string(iter));
        }
        iter++;
    }
}
void DiscriminativeModel::predict(dataset &ds)
{
    batchPredict(ds, 0, ds.n);
    ds.y_predict.resize(ds.n, 1);
    int maxpos; // Correct type for indexing
    for (int i = 0; i < ds.n; i++)
    {
        maxpos = 0;
        for (int j = 1; j < K; j++)
        {
            if (ds.y_k[i][j] > ds.y_k[i][maxpos])
            {
                maxpos = j;
            }
        }
        ds.y_predict[i][0] = maxpos;
    }
}
void DiscriminativeModel::eval(dataset &ds)
{
    predict(ds);
    ds.k = K;
    ds.ConfusionMatrix();
    // cout << "\033[1;33mConfusion ";
    // ds.confusion_matrix.print();
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
    cout << setw(10) << train.accuracy[0];
    cout << " \033[35m<<< Valid <<< \033[0m";
    eval(valid);
    cout << setw(10) << valid.accuracy[0];
    cout << endl;
}
void DiscriminativeModel::rename(const std::string &modelName)
{
    name = modelName;
}
void DiscriminativeModel::load(const std::string &modelName, const std::string &pre)
{
    std::string _w = pre + modelName + ".csv";
    std::string _set = pre + modelName + "_param.csv";
    w.read(_w);
    matrix tmp;
    tmp.read(_set);
    K = (int)tmp[0][0];
    M = (int)tmp[0][1];
}
void DiscriminativeModel::load(const std::string &modelName)
{
    load(modelName, "model/");
}
void DiscriminativeModel::save(const std::string &modelName, const std::string &pre)
{
    std::string _w = pre + modelName + ".csv";
    std::string _p = pre + modelName + "_param.csv";
    std::string _s = pre + modelName + "_setting.csv";
    // weights
    w.save(_w);
    // K, M
    matrix tmp(1, 2);
    tmp[0][0] = K;
    tmp[0][1] = M;
    tmp.save(_p);
    tmp.resize(1, 3);
    tmp[0][0] = lr;
    tmp[0][1] = batch_size;
    tmp[0][2] = epoch;
    tmp.save(_s);
}
void DiscriminativeModel::save(const std::string &modelName)
{
    save(modelName, "model/");
}
void DiscriminativeModel::save()
{
    save(name);
}
void DiscriminativeModel::saveLog(const std::string &filename)
{
    update_log.save("homework/"+filename+"_log.csv");
}
void DiscriminativeModel::saveLog()
{
    saveLog(name);
}