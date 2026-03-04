/* Matrix helpers used by symnmf: allocation, distance, multiplication, etc. */
#include "matrix_ops.h"
#include <stdlib.h>
#include <math.h>

double **allocate_matrix(int rows, int cols)
{
    int i;
    double **M = (double **)malloc((size_t)rows * sizeof(double *));
    if (M == NULL) return NULL;
    for (i = 0; i < rows; i++) {
        M[i] = (double *)malloc((size_t)cols * sizeof(double));
        if (M[i] == NULL) {
            while (i-- > 0) free(M[i]);
            free(M);
            return NULL;
        }
    }
    return M;
}

void free_matrix(double **M, int rows)
{
    int i;
    if (M == NULL) return;
    for (i = 0; i < rows; i++)
        free(M[i]);
    free(M);
}

double squared_euclidean_dist(const double *a, const double *b, int d)
{
    double sum = 0.0;
    double diff;
    int i;
    for (i = 0; i < d; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/* C = A*B (A n×m, B m×p, C n×p). */
void mat_mult(double **A, double **B, double **C, int n, int m, int p)
{
    int i, j, t;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (t = 0; t < m; t++)
                C[i][j] += A[i][t] * B[t][j];
        }
    }
}

/* C = A^T * B (for H^T*H in the update). */
void mat_mult_ATB(double **A, double **B, double **C, int n, int m, int p)
{
    int i, j, t;
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (t = 0; t < n; t++)
                C[i][j] += A[t][i] * B[t][j];
        }
    }
}

/* Sum of (A[i][j] - B[i][j])^2; used for convergence check. */
double frobenius_sq_diff(double **A, double **B, int n, int m)
{
    double sum = 0.0;
    double d;
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++) {
            d = A[i][j] - B[i][j];
            sum += d * d;
        }
    return sum;
}

void copy_matrix(double **src, double **dst, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            dst[i][j] = src[i][j];
}
