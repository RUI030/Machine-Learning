# Linear Regression
> [!CAUTION]
> Makefile and the model itself are not finished yet.

## Run this code
* Compile:
```
g++ -std=c++11 -o main main.cpp regression.cpp dataset.cpp matrix.cpp
```
* Run:
```
./main
```
## Matrix
* **`clear`** : clear the matrix
* **`print`** : print the matrix in the terminal
* **`copy`** : copy from another matrix
* **`append`** : append a new row or multiple rows
* **`concat`** : concatenate the matrix with another one
* **`fill`** : fill the matrix with given value
* **`scale`** : scale each element with given scalar
* **`split`** : create a submatrix from another matrix given the range of row and column indecies
* **`add`** / **`sub`** : add/sub the matrix itself with another matrix
* **`dot`** : inner product the matrix itself with another matrix
* **`update`** : update the mean value and standard deviation of each column
* **`normalize`** : normalize the matrix by column (or given another group of mean and standard deviation)
> [!NOTE]  
> **Todo:**
> * operation overload
> * LU decomposition, ..., and alot :(
## Dataset
### Function:

## Regression


