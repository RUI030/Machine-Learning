CC = g++ -std=c++11

main:	main.o	dataset.o
    ${CC}	main.o dataset.o	-o	main

main.o:	main.cpp
    ${CC}	main.cpp	-c

dataset.o:	dataset.cpp
    ${CC}	dataset.cpp	-c

clean:
    rm -rf	main.o	dataset.o
