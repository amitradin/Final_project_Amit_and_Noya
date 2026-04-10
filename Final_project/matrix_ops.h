#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

/* This function is for allocating a matrix */
double **allocate_matrix(int rows, int cols);
/* This function is for freeing a matrix */
void free_matrix(double **M, int rows);
/* This function is for computing the squared Euclidean distance between two vectors */
double squared_euclidean_dist(const double *a, const double *b, int d);
/* This function is for multiplying two matrices */
void mat_mult(double **A, double **B, double **C, int n, int m, int p);
/* This function is for multiplying two matrices */
void mat_mult_ATB(double **A, double **B, double **C, int n, int m, int p);
/* This function is for computing the Frobenius square difference between two matrices */
double frobenius_sq_diff(double **A, double **B, int n, int m);
/* This function is for copying a matrix */
void copy_matrix(double **src, double **dst, int rows, int cols);

#endif
