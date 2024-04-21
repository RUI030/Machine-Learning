#include "dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// Constructor
dataset::dataset()
{
    clear();
}
// Destructor
dataset::~dataset() {}
void dataset::read(const std::string &filename, int y_dim)
{
    std::ifstream fin(filename);
    std::string line, word;

    // Read the header
    if (std::getline(fin, line))
    {
        std::stringstream s(line);
        while (std::getline(s, word, ','))
        {
            if (y.header.size() < y_dim)
            {
                y.header.push_back(word);
            }
            else
            {
                x.header.push_back(word);
            }
        }
    }
    k = x.header.size() + y.header.size();
    ydim = y.header.size();
    xdim = x.header.size();

    // Read the data
    while (std::getline(fin, line))
    {
        std::stringstream s(line);
        std::vector<double> xRow, yRow;
        int colIndex = 0;
        while (std::getline(s, word, ','))
        {
            double value = std::stod(word);
            if (colIndex < ydim)
            {
                yRow.push_back(value);
            }
            else
            {
                xRow.push_back(value);
            }
            colIndex++;
        }
        y.append(yRow);
        x.append(xRow);
        n++;
    }
    fin.close();
    std::cout << "\033[1;32mSuccessfully read the file [ " << filename << " ] !\033[0m" << std::endl;
}
void dataset::read(const std::string &filename)
{
    // Default readFile function with y_dim = 1
    read(filename, 1);
}
void dataset::printRow(int i)
{
    std::cout << "\033[1;33m";
    y.printRow(i);
    std::cout << "\033[0m  |  ";
    x.printRow(i, 7);
    std::cout << std::endl;
}
void dataset::print()
{
    std::cout << "\nDataset: ";
    std::cout << "\033[1;33mY:( " << y.row() << " x " << y.col() << " )";
    std::cout << "\t\033[0mX:( " << x.row() << " x " << x.col() << " )\n"
              << std::endl;
    for (int i = 0; i < std::min(n, 12); i++)
    {
        printRow(i);
    }
    if (n > 12)
        std::cout << "\n\t\t... other rows are omitted ...\n"
                  << std::endl;
}
void dataset::copy(const dataset &source, int r1, int r2)
{
    n = r2 - r1;
    k = source.k;
    xdim = source.xdim;
    ydim = source.ydim;
    x.slice(source.x, r1, r2);
    y.slice(source.y, r1, r2);
}
void dataset::copy(const dataset &source)
{
    copy(source, 0, source.n);
}
void dataset::split(dataset &train, dataset &valid, int idx)
{
    if (idx < 0 || idx >= n)
    {
        std::cerr << "Invalid split index: " << idx << std::endl;
        return;
    }
    train.copy(*this, 0, idx);
    valid.copy(*this, idx, n);
    train.update();
    valid.update();
}
void dataset::T()
{
    x.T();
    y.T();
}
void dataset::update()
{
    x.update();
    y.update();
}
void dataset::normalize(const std::vector<double> m, const std::vector<double> s, const std::vector<double> m_y, const std::vector<double> s_y)
{
    x.normalize(m, s);
    y.normalize(m_y, s_y);
}
void dataset::normalize(const dataset &source)
{
    x.normalize(source.x);
    y.normalize(source.y);
}
void dataset::normalize()
{
    x.normalize();
    y.normalize();
}
void dataset::save(const std::string &filename)
{
    // todo
}
void dataset::clear()
{
    x.clear();
    y.clear();
    y_predict.clear();
    k=0;
    xdim=0;
    ydim=0;
    n=0;
    PHI.clear();
    accuracy.clear();
    MSE.clear();
    confusion_matrix.clear();
}
void dataset::designMatrix(const int M, const double s, const std::vector<double> &u)
{
    int N = x.row(); // number of row of data
    int K = x.col(); // number of features
    // resize design matrix
    PHI.resize(N, K * M);
    // calculate design matrix
    for (int i = 0; i < N; i++) // number of rows
    {
        for (int k = 0; k < K; k++) // concatenate the design matrix of different feature
        {
            PHI[i][k * M] = 1; // phi(x) = 1 while j = 0 ==> redundant item for convenience
            for (int j = 1; j < M; j++)
            {
                PHI[i][k * M + j] = 1 / (1 + exp(-(x[i][k] - u[j]) / s)); // note the negative sign in the exponent
            }
        }
    }
}
void dataset::ConfusionMatrix()
{
    // Calculate the confusion matrix
    int y_true;
    int y_pred;
    confusion_matrix.resize(4, 4, 0);
    for (int i = 0; i < n; i++)
    {
        y_true = (int)y[i][0];
        y_pred = (int)y_predict[i][0];
        confusion_matrix[y_true][y_pred]++;
    }
}