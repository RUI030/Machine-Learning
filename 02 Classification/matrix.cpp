#include "matrix.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>

using namespace std;

matrix::matrix(int rows, int cols)
{
    resize(rows, cols);
}
matrix::matrix()
{
    matrix(0, 0);
}
matrix::~matrix()
{
    clear();
}
// Element access
double &matrix::at(int row, int col)
{
    if (row < 0 || row >= r || col < 0 || col >= c)
    {
        throw std::out_of_range("matrix::at(): index out of range");
    }
    return data[row][col];
}
const double &matrix::at(int row, int col) const
{
    if (row < 0 || row >= r || col < 0 || col >= c)
    {
        throw std::out_of_range("matrix::at() const: index out of range");
    }
    return data[row][col];
}
std::vector<double> &matrix::operator[](int row)
{
    return data[row];
}
const std::vector<double> &matrix::operator[](int row) const
{
    return data[row];
}
// Operator overloads
matrix &matrix::operator=(const matrix &rhs)
{
    if (this != &rhs)
    {
        copy(rhs);
    }
    return *this;
}
matrix matrix::operator+(const matrix &rhs) const
{
    matrix result = *this;
    result.add(rhs);
    return result;
}
matrix matrix::operator*(const matrix &rhs) const
{
    matrix result = *this;
    result.dot(rhs);
    return result;
}
matrix matrix::operator*(double scalar) const
{
    matrix result = *this;
    result.scale(scalar);
    return result;
}
matrix matrix::operator*(float scalar) const
{
    matrix result = *this;
    result.scale((double)scalar);
    return result;
}
matrix matrix::operator*(int scalar) const
{
    matrix result = *this;
    result.scale((double)scalar);
    return result;
}
bool matrix::empty()
{
    return data.empty();
}
int matrix::row() const { return r; }
int matrix::col() const { return c; }
std::string matrix::dim() const
{
    return "( " + std::to_string(r) + " x " + std::to_string(c) + " )";
}
double matrix::det()
{
    calcDet();
    return Det;
}
void matrix::resize(int newRows, int newCols, double fillValue)
{
    // Resize each row to the new column size without changing existing values
    for (auto &row : data)
    {
        if (row.size() < newCols)
        {
            row.resize(newCols, fillValue);
        }
    }

    // Add new rows if the new row count is greater than the current row count
    while (data.size() < newRows)
    {
        data.emplace_back(newCols, fillValue);
    }

    // Update the number of rows and columns
    r = newRows;
    c = newCols;
}
void matrix::append(const vector<double> &newRow)
{
    if (!data.empty() && c != newRow.size())
    {
        cout << "\033[1;31m[FAIL]: mismatched column size.\033[0m"
             << "\tmatrix:" << c << "\tnew row:" << newRow.size() << endl;
        return;
    }
    data.emplace_back(newRow);
    r++;
    c = newRow.size();
}
void matrix::append(const matrix &source, int r1, int r2)
{
    if (data.empty())
    {
        r = 0;
        c = source.col();
    }
    else if (c != source.col())
    {
        cout << "\033[1;31m[FAIL]: mismatched column size.\033[0m"
             << "\tThis:" << c << "\tSource:" << source.col() << endl;
        return;
    }
    if (r1 > r2)
        swap(r1, r2);
    if (r1 < 0 || r2 >= source.row())
    {
        cout << "\033[1;31m[FAIL]: Index out of bounds.\033[0m" << endl;
        return;
    }
    for (int i = r1; i <= r2; i++)
    {
        data.emplace_back(source.data[i]);
    }
    r += r2 - r1 + 1;
}
void matrix::append(const matrix &source)
{
    append(source, 0, source.row() - 1);
}
void matrix::concat(const matrix &source, int c1, int c2)
{
    if (data.empty())
    {
        r = source.row();
        c = 0;
        for (int i = 0; i < r; i++)
        {
            std::vector<double> newRow;
            data.emplace_back(newRow);
        }
    }
    else if (r != source.row())
    {
        cout << "\033[1;31m[FAIL]: mismatched column size.\033[0m"
             << "\tThis:" << r << "\tSource:" << source.row() << endl;
        return;
    }
    if (c1 > c2)
        swap(c1, c2);
    if (c1 < 0 || c2 >= source.col())
    {
        cout << "\033[1;31m[FAIL]: Index out of bounds.\033[0m" << endl;
        return;
    }
    for (int i = 0; i < r; i++)
    {
        data[i].insert(data[i].end(), source.data[i].begin() + c1, source.data[i].begin() + c2);
    }
    c += c2 - c1 + 1;
    header.insert(header.end(), source.header.begin() + c1, source.header.begin() + c2);
}
void matrix::concat(const matrix &source)
{
    concat(source, 0, source.col() - 1);
}
void matrix::calcMean()
{
    double m;
    Mean.resize(data[0].size());
    for (int j = 0; j < c; j++)
    {
        m = 0;
        for (int i = 0; i < r; i++)
        {
            m += data[i][j];
        }
        Mean[j] = m / r;
    }
}
void matrix::calcSTD()
{
    STD.resize(data[0].size());
    if (Mean.empty())
        calcMean();
    double sd;
    for (int j = 0; j < c; j++)
    {
        sd = 0;
        for (int i = 0; i < r; i++)
        {
            sd += pow(data[i][j] - Mean[j], 2);
        }
        sd /= (r - 1);
        sd = sqrt(sd);
        STD[j] = sd;
    }
}
void matrix::update()
{
    calcMean();
    calcSTD();
}
void matrix::copy(const matrix &source)
{
    r = source.r;
    c = source.c;
    data = source.data;
    Mean = source.Mean;
    STD = source.STD;
}
void matrix::clear()
{
    header.clear();
    Mean.clear();
    STD.clear();
    r = 0;
    c = 0;
    Det = 0;
    for (auto &rowi : data)
    {
        rowi.clear();
    }
    data.clear();
}
void matrix::I(const int d, const double val)
{
    r = d;
    c = d;
    resize(d, d);
    for (int i = 0; i < d; i++)
        data[i][i] = val;
}
void matrix::I(const int d)
{
    I(d, 1.0);
}
void matrix::range(const double start, const double stop, const int step)
{
    resize(1, step);
    double a, b;
    a = start;
    b = (stop - start) / (double)step;
    for (int i = 0; i < step; i++)
    {
        data[0][i] = a;
        a += b;
    }
}
void matrix::T()
{
    vector<vector<double>> newData(c, vector<double>(r));
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            newData[j][i] = data[i][j];
        }
    }
    data = newData;
    swap(r, c);
}
void matrix::normalize(const std::vector<double> m, const std::vector<double> s)
{
    if (m.size() != c || s.size() != c)
        cout << "\033[1;31m[FAIL]: mismatched row number.\033[0m This:" << r << " Mean:" << m.size() << " STD:" << s.size() << endl;
    for (int j = 0; j < c; j++)
    {
        for (int i = 0; i < r; i++)
        {
            data[i][j] = (data[i][j] - m[j]) / s[j];
        }
    }
    update();
}
void matrix::normalize(const matrix &source)
{
    normalize(source.Mean, source.STD);
}
void matrix::normalize()
{
    if (STD.empty())
        calcSTD();
    normalize(Mean, STD);
}
void matrix::slice(const matrix &source, int r1, int r2, int c1, int c2)
{
    if (r1 > r2)
        swap(r1, r2);
    if (c1 > c2)
        swap(c1, c2);
    if (r1 < 0 || c1 < 0 || r2 > source.row() || c2 > source.col())
    {
        cout << "\033[1;31m[FAIL]: Index out of bounds.\033[0m" << endl;
        return;
    }
    this->clear();
    r = r2 - r1 ;
    c = c2 - c1 ;
    data.reserve(r);
    for (int i = r1; i < r2; i++)
    {
        vector<double> newRow(source.data[i].begin() + c1, source.data[i].begin() + c2 );
        this->data.emplace_back(newRow);
    }
}
void matrix::slice(const matrix &source, int r1, int r2)
{
    slice(source, r1, r2, 0, source.col());
}
void matrix::fill(double val)
{
    for (int i = 0; i < r; i++)
    {
        std::fill(data[i].begin(), data[i].end(), val);
    }
}
void matrix::add(const matrix &source)
{
    if (r != source.r || c != source.c)
    {
        cout << "\033[1;31m[FAIL]: mismatched dimensions.\033[0m" << endl;
        return;
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            data[i][j] += source.data[i][j];
        }
    }
}
void matrix::sub(const matrix &source)
{
    if (r != source.r || c != source.c)
    {
        cout << "\033[1;31m[FAIL]: mismatched dimensions.\033[0m" << endl;
        return;
    }
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            data[i][j] -= source.data[i][j];
        }
    }
}
void matrix::dot(const matrix &source)
{
    if (c != source.row())
    {
        cout << "\033[1;31m[FAIL]: mismatched dimensions for matrix multiplication. " << dim() << " . " << source.dim() << "\033[0m" << endl;
        return;
    }
    vector<vector<double>> result(r, vector<double>(source.c, 0));
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < source.c; j++)
        {
            for (int k = 0; k < c; k++)
            {
                result[i][j] += data[i][k] * source.data[k][j];
            }
        }
    }
    data = result;
    c = source.col();
}
void matrix::scale(double val)
{
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            data[i][j] *= val;
        }
    }
}
void matrix::calcDet()
{
    if (r != c)
    {
        cout << "\033[1;31m[FAIL]: Determinant can only be calculated for square matrices.\033[0m" << endl;
        Det = 0;
        return;
    }

    int n = r;
    Det = 1.0;
    vector<vector<double>> lu(n, vector<double>(n));

    // Perform LU decomposition
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < j; ++k)
            {
                sum += lu[i][k] * lu[k][j];
            }
            lu[i][j] = data[i][j] - sum;
        }

        for (int j = i + 1; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum += lu[i][k] * lu[k][j];
            }
            lu[i][j] = (data[i][j] - sum) / lu[i][i];
        }
    }

    // Calculate determinant from LU decomposition
    for (int i = 0; i < n; ++i)
    {
        Det *= lu[i][i];
    }
}
void matrix::inv()
{
    if (r != c)
    {
        cout << "\033[1;31m[FAIL]: Inverse can only be calculated for square matrices.\033[0m" << endl;
        return;
    }

    int n = r;
    vector<vector<double>> lu(n, vector<double>(n));
    vector<vector<double>> inv(n, vector<double>(n));

    // Perform LU decomposition
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < j; ++k)
            {
                sum += lu[i][k] * lu[k][j];
            }
            lu[i][j] = data[i][j] - sum;
        }

        for (int j = i + 1; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum += lu[i][k] * lu[k][j];
            }
            lu[i][j] = (data[i][j] - sum) / lu[i][i];
        }
    }

    // Solve for the inverse using forward and backward substitution
    for (int i = 0; i < n; ++i)
    {
        vector<double> e(n, 0);
        e[i] = 1;
        vector<double> y(n, 0);

        // Forward substitution for Ly = e
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < j; ++k)
            {
                sum += lu[j][k] * y[k];
            }
            y[j] = (e[j] - sum) / lu[j][j];
        }

        // Backward substitution for Ux = y
        for (int j = n - 1; j >= 0; --j)
        {
            double sum = 0.0;
            for (int k = j + 1; k < n; ++k)
            {
                sum += lu[j][k] * inv[k][i];
            }
            inv[j][i] = (y[j] - sum) / lu[j][j];
        }
    }

    // Copy the result back to the original matrix
    data = inv;
}
void matrix::abs()
{
    for (int i = 0; i < row(); i++)
    {
        for (int j = 0; j < col(); j++)
        {
            data[i][j] = (data[i][j] < 0) ? -data[i][j] : data[i][j];
        }
    }
}
void matrix::svd(matrix &U, matrix &Sigma, matrix &V) const
{
    Eigen::MatrixXd eigMat(r, c);
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            eigMat(i, j) = data[i][j];
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U_eig = svd.matrixU();
    Eigen::MatrixXd Sigma_eig = svd.singularValues().asDiagonal();
    Eigen::MatrixXd V_eig = svd.matrixV();

    U.resize(U_eig.rows(), U_eig.cols());
    for (int i = 0; i < U_eig.rows(); ++i)
    {
        for (int j = 0; j < U_eig.cols(); ++j)
        {
            U.data[i][j] = U_eig(i, j);
        }
    }

    Sigma.resize(Sigma_eig.rows(), Sigma_eig.cols());
    for (int i = 0; i < Sigma_eig.rows(); ++i)
    {
        for (int j = 0; j < Sigma_eig.cols(); ++j)
        {
            Sigma.data[i][j] = Sigma_eig(i, j);
        }
    }

    V.resize(V_eig.rows(), V_eig.cols());
    for (int i = 0; i < V_eig.rows(); ++i)
    {
        for (int j = 0; j < V_eig.cols(); ++j)
        {
            V.data[i][j] = V_eig(i, j);
        }
    }
}
void matrix::designMatrix(matrix &PHI, const int M, const double s, const std::vector<double> &u)
{
    int N = row(); // number of row of data
    int K = col(); // number of features
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
                PHI[i][k * M + j] = 1 / (1 + exp(-(data[i][k] - u[j]) / s)); // note the negative sign in the exponent
            }
        }
    }
}
void matrix::printRow(int ri, int l)
{
    if (ri < 0 || ri >= r)
    {
        cout << "\033[1;31m[FAIL]: Row index out of bounds.\033[0m";
        return;
    }
    for (int j = 0; j < min(c, l); j++)
    {
        cout << setw(15) << data[ri][j];
    }
    if (c > l)
    {
        cout << "     ...     " << data[ri][c - 1];
    }
}
void matrix::printRow(int ri)
{
    printRow(ri, 8);
}
void matrix::print()
{
    std::cout << "\033[1;33mMatrix: ";
    std::cout << "( " << this->row() << " x " << this->col() << " )\033[0m\n\n";
    for (int i = 0; i < min(r, 12); i++)
    {
        printRow(i);
        cout << endl;
    }
    if (r > 12)
        cout << "\n\t\t... other rows are omitted\n"
             << endl;
}
void matrix::stat()
{
    cout << "=======================================" << endl;
    cout << "\n\033[1m\033[35mMean:\033[0m" << endl;
    for (int j = 0; j < Mean.size(); j++)
    {
        cout << setw(10) << Mean[j];
    }
    cout << "\n\n\033[1m\033[35mSTD:\033[0m" << endl;
    for (int j = 0; j < STD.size(); j++)
    {
        cout << setw(10) << STD[j];
    }
    cout << "\n\n=======================================\n"
         << endl;
}
void matrix::read(const std::string &filename)
{
    std::ifstream file(filename);
    std::string line;
    data.clear();
    r = 0;
    c = 0;
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ','))
        {
            try
            {
                row.push_back(std::stod(value));
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Error: Invalid argument in row " << r + 1 << ", column " << row.size() + 1 << std::endl;
                file.close();
                return;
            }
            catch (const std::out_of_range &e)
            {
                std::cerr << "Error: Out of range value in row " << r + 1 << ", column " << row.size() + 1 << std::endl;
                file.close();
                return;
            }
        }
        if (c == 0)
        {
            c = row.size();
        }
        else if (c != row.size())
        {
            std::cerr << "Error: Inconsistent number of columns in row " << r + 1 << std::endl;
            file.close();
            return;
        }
        data.push_back(row);
        ++r;
    }

    file.close();
}
void matrix::save(const std::string &filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    for (const auto &row : data)
    {
        for (size_t i = 0; i < row.size(); ++i)
        {
            file << row[i];
            if (i < row.size() - 1)
            {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}
// reshape: preserve the element while resizing

// Eigen tutorial
// https://www.youtube.com/watch?v=fUxp3upZsk0&ab_channel=AleksandarHaber