#ifndef HELPERS_H
#define HELPERS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

int OneSample(const arma::vec& vw);
int OneSample_uniform(const int size);
double log_beta(const double a, const double b);
double log_sum_vec(const vec& log_x);
double log_sum_mat(const mat& log_x);
vec log_normalize_vec(const vec& log_x);
vec normalize_vec(const vec& x);
mat log_normalize_mat(const mat& log_x);
ivec convert_vector_to_ivec(vector<int> vector);
ivec get_subvector_int(ivec x, vector<int> indices, int size);
vec get_subvector_double(vec x, vector<int> indices, int size);
mat get_submat_double(mat x, vector<int> indices, int size);

double logit(double x);
double logit_inv(double x);

mat rotation_matrix(int n);

int random_number(int size);

double rinversegauss_single(double mu, double lambda);


#endif