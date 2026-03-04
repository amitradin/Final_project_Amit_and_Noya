#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

/* Matrix allocation, distance, multiply, copy. */
double **allocate_matrix(int rows, int cols);
void free_matrix(double **M, int rows);
double squared_euclidean_dist(const double *a, const double *b, int d);
void mat_mult(double **A, double **B, double **C, int n, int m, int p);
void mat_mult_ATB(double **A, double **B, double **C, int n, int m, int p);
double frobenius_sq_diff(double **A, double **B, int n, int m);
void copy_matrix(double **src, double **dst, int rows, int cols);

#endif
