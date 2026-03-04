#ifndef SYMNMF_H
#define SYMNMF_H

/* SymNMF core: similarity A, degree D, normalized W, and H update. */
double **sym(const double **points, int n, int d);
double **ddg(const double **points, int n, int d);
double **norm(const double **points, int n, int d);
void symnmf_run(double **W, double **H, int n, int k, double epsilon, int max_iter);

#endif
