# Linear Regression
> [!CAUTION]
> Makefile and the model itself are not finished yet.

## Run this code
> [!CAUTION]
> It's not working now :(
> Please [compile it with `cl`](#using-cl-to-compile-the-project) in VSCode
* Compile:
```
g++ -std=c++11 -o main main.cpp regression.cpp dataset.cpp matrix.cpp
```
* Run:
```
./main
```
## Dependency
This project use [**eigen**](https://eigen.tuxfamily.org/index.php?title=Main_Page) for *Singular Value Decomposition* (SVD), you can follow the steps in the [video](https://www.youtube.com/watch?v=fUxp3upZsk0&ab_channel=AleksandarHaber) to install the library for vscode users.
1. [Download eigen here](https://eigen.tuxfamily.org/index.php?title=Main_Page)
2. Unzip the file to the path you want, for example I put it at
```
C:\toolbox
```
3. Make sure you are able to [compile the code in vscode](#using-cl-to-compile-the-project) with `cl` .
4. [Open VSCode with command `code.`](#open-vscode-with-command-code)
5. Press the triangle button (Run C/C++) at the top right of the window
6. Find the folder **`.vscode`** which should be in your project directory
7. Open **`task.json`** and modify
8. Press **`Ctrl`** + **`Shift`** + **`P`**, search **`C/C++ Edit Configuration (UI)`**
9. Open **`c_cpp_properties.json`** in the folder **`.vscode`** and modify it
## Adding Environment `Path`
1. open **`Setting`** `設定` (or directly search the stuff in step ii)
2. search **`Edit environment variables for your account`** (the search bar should be at the upper left corner of the window you opened from step i) `編輯您的帳戶的環境變數` (直接找 `環境變數` 應該就能找到)
3. find and click on **`Path`** in `Users variable` `使用者環境變數`
4. click **`Edit`** `編輯(E)`, then click **`New`** `新增(N)`
5. type , press **`Enter`**, and  press every **`OK`**.
   * MinGW (default): 
```
C:\msys64\ucrt64\bin
```
   * VSCode `code.` (example):
```
C:\Users\RUI\AppData\Local\Programs\Microsoft VS Code\bin
```
## Debugging
### using `cl` to compile the project
> Reference: [compile the code in vscode](https://code.visualstudio.com/docs/languages/cpp)
1. install the C/C++ plug-in in VSCode

2. install [**MinGW**](https://www.mingw-w64.org/downloads/) with the [instructions](https://code.visualstudio.com/docs/cpp/config-mingw#_create-a-hello-world-app). Please make sure the host correspond to your OS, for example I use Windows 11, and I installed [MSYS2](https://www.msys2.org/)

3. after installation, find **`MSYS2 MINGW64`** in the search bar in your computer (the place you search for software applications :) ) and past the following command into the terminal:
```
pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain
```
  press **`Enter`** or **`Y`** while needed :)

4. add environment path (the path that you installed MinGW) to your computer. check [here](#adding-environment-path) for detailed instructions. default path should be: 
```
C:\msys64\ucrt64\bin
```
5. search `cmd` in the search bar then paste the following command to check if you successfully installed MinGW
```
gcc --version
g++ --version
gdb --version
```
### Open VSCode with command `code`
1. open `Developer Command Prompt for VS 2022`, us the command `cd {your project path}` to the place that you want to code, ex:
(please remember to check if the username in the file path is correct!!)
```
cd C:\Users\RUI\Document\Coding\MyProject
```
2. type the command `code .` to open VSCode so that you can use `cl` to compile your project 
```
code .
``` 
### `code .` not working
* [Adding the environment path](#adding-environment-path), the default path should look like:
(remember to change the username)
```
C:\Users\RUI\AppData\Local\Programs\Microsoft VS Code\bin
```
### still can not compile
* Install [**`Windows SDK`**](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)

## Matrix
* **`clear`** : clear the matrix
* **`print`** : print the matrix in the terminal
* **`copy`** : copy from another matrix
* **`append`** : append a new row or multiple rows
* **`concat`** : concatenate the matrix with another one
* **`fill`** : fill the matrix with given value
* **`scale`** : scale each element with given scalar
* **`slice`** : create a submatrix from another matrix given the range of row and column indecies
* **`add`** / **`sub`** : add/sub the matrix itself with another matrix
* **`dot`** : inner product the matrix itself with another matrix
* **`update`** : update the mean value and standard deviation of each column
* **`normalize`** : normalize the matrix by column (or given another group of mean and standard deviation)
### operator
* **`=`** : copy the matrix from the rhs.
* **`+`** : adding 2 matrices. NOT CONCAT!!!!!! (the two matrix should have the same dimension)
* **`*`** : the inner product of 2 matrices
> [!NOTE]  
> **Todo:**
> * a lot of Linear algebra stuff :(
## Dataset
### Function:

## Regression


