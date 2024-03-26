#include <iostream>
#include <string>
#include <vector>
#include "regression.h"

using namespace std;

int main()
{
    int k = 5, maxM = 20, step = 1;
    dataset hw1, sub[5], test;
    LinearRegression model;
    double _mse[30][30], _acc[30][30], _acc_, _mse_; //[m][fold]
    vector<double> mse, acc, M;
    hw1.read("HW1.csv"); // load data
    // split into K subset
    int cut[6];    // index to split the subset
    int n = 15000; // first 1000 for training and cross validation
    cut[0] = 0;
    cut[5] = n;
    for (int c = 1; c <= k; c++)
    {
        cut[c] = c * n / k;
        sub[c - 1].copy(hw1, cut[c - 1], cut[c]);
    }
    test.copy(hw1, n, hw1.n);
    hw1.clear();
    // Note: it's time consuming to load and preprocess the data alot of time
    // So maybe fix data and let m change is better and faster
    for (int c = 0; c < k; c++)
    {
        cout << "\033[1;34m[FOLD]:" << c + 1 << "\033[0m" << endl;
        // clear
        model.train.x.clear();
        model.train.y.clear();
        model.valid.x.clear();
        model.valid.y.clear();
        // put the data in
        for (int d = 0; d < k; d++)
        {
            if (d == c)
            {
                model.valid.x.append(sub[d].x);
                model.valid.y.append(sub[d].y);
            }
            else
            {
                model.train.x.append(sub[d].x);
                model.train.y.append(sub[d].y);
            }
        }
        // preprocessing
        model.prep();
        for (int m = 3; m <= maxM; m += step)
        {
            model.setting(m);
            // train
            model.update();
            // evaluate
            model.eval();
            // record Loss Function, which could calculate from MSE since the given error function = 1/2 SUM((y-t)^2) = Nd/2 * MSE
            _mse[m][c] = model.valid.MSE[0];
            _acc[m][c] = model.valid.accuracy[0];
        }
    }
    for (int m = 3; m <= maxM; m += step)
    {
        _mse_ = 0.0;
        _acc_ = 0.0;
        for (int c = 0; c < k; c++)
        {
            _mse_ += _mse[m][c];
            _acc_ += _acc[m][c];
        }
        _mse_/=(double)k;
        _acc_/=(double)k;
        mse.push_back(_mse_);
        acc.push_back(_acc_);
    }
    // save result
    matrix res;
    res.append(M);
    res.append(mse);
    res.append(acc);
    res.save("homework/Q3/res.csv");
    // Choose best Model
    double minMSE = 10000, maxACC = -100000, mMSE, mACC;
    for (int i = 0; i < M.size(); i++)
    {
        if (mse[i] < minMSE)
        {
            minMSE = mse[i];
            mMSE = M[i];
        }
        if (acc[i] > maxACC)
        {
            maxACC = acc[i];
            mACC = M[i];
        }
    }
    // print result
    cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
    cout << " max [ACC]: " << maxACC << " at M = " << mACC << endl;
    cout << " min [MSE]: " << minMSE << " at M = " << mMSE << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    // Train & save model
    return 0;
}
