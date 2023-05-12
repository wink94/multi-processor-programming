#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 100

void Add_Matrix(int** matrix1, int** matrix2, int** result, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

int addMatrixMain() {
    int** matrix1 = (int**)malloc(MATRIX_SIZE * sizeof(int*));
    int** matrix2 = (int**)malloc(MATRIX_SIZE * sizeof(int*));
    int** result = (int**)malloc(MATRIX_SIZE * sizeof(int*));

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        matrix1[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
        matrix2[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
        result[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
    }

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            matrix1[i][j] = i * j;
            matrix2[i][j] = i + j;
        }
    }

    clock_t start = clock();
    Add_Matrix(matrix1, matrix2, result, MATRIX_SIZE);
    clock_t end = clock();

    double execution_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", execution_time);

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        free(matrix1[i]);
        free(matrix2[i]);
        free(result[i]);
    }

    free(matrix1);
    free(matrix2);
    free(result);

    return 0;
}
