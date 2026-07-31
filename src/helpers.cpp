// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

int OneSample(const arma::vec& vw){
  double u = R::runif(0,1);
  arma::uvec out = find(cumsum(vw) > u, 1);
  
  return (int) out(0);
}

int OneSample_uniform(const int size){
  vec vw(size);
  vw.fill(1.0 / (double) size);
  return OneSample(vw);
}

double log_beta(const double a, const double b)
{
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

double log_sum_vec(const vec& log_x){
  double log_x_max = log_x.max();
  return log_x_max + log(sum(exp(log_x - log_x_max)));
}

double log_sum_mat(const mat& log_x){
  return log_sum_vec(vectorise(log_x));
}

vec log_normalize_vec(const vec& log_x){
  return exp(log_x - log_sum_vec(log_x));
}

vec normalize_vec(const vec& x){
  double sum_temp = sum(x);
  return x / sum_temp;
}

mat log_normalize_mat(const mat& log_x){
  int n_rows = log_x.n_rows;
  int n_cols = log_x.n_cols;
  return reshape(log_normalize_vec(vectorise(log_x)), n_rows, n_cols);
}

ivec convert_vector_to_ivec(vector<int> vector){
  int size = vector.size();
  ivec out(size);
  
  for(int i=0; i<size; i++){
    out(i) = vector[i];
  }
  
  return out;
}

ivec get_subvector_int(ivec x, vector<int> indices, int size){
  ivec out(size);
  for(int i=0; i<size; i++){
    int index = indices[i];
    out(i) = x(index);
  }
  return out;
}


vec get_subvector_double(vec x, vector<int> indices, int size){
  vec out(size);
  for(int i=0; i<size; i++){
    int index = indices[i];
    out(i) = x(index);
  }
  return out;
}

mat get_submat_double(mat x, vector<int> indices, int size){
  int dim = x.n_rows;
  mat out(dim, size);
  for(int i=0; i<size; i++){
    int index = indices[i];
    out.col(i) = x.col(index);
  }
  return out;
}

double logit(double x){
  return log(x) - log(1.0-x);
}

double logit_inv(double x){
  return exp(x) / (1+exp(x));
}

// [[Rcpp::export]]
mat rotation_matrix(int n){
  mat M(n,n);
  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      M(i,j) = R::rnorm(0.0, 1.0);
    }
  }
  
  mat Q;
  mat R;
  
  qr(Q, R, M);
  
  for(int i=0; i<n; i++){
    if(R(i,i) < 0){
      Q.col(i) = - 1.0 * Q.col(i);
    }
  }
  
  if(det(Q) < 0.0){
    Q.col(0) = - 1.0 * Q.col(0);
  }
  
  return Q;
}

int random_number(int size){
  int num = arma::randi<int>(arma::distr_param(0, size-1));
  return num;
}

// function to sample from the inverse Gaussian
double rinversegauss_single(double mu, double lambda){
  
  double nu = R::rnorm(0.0, 1.0);
  double y = pow(nu, 2.0);
  double mu2 = pow(mu, 2.0);
  
  double x = mu + mu2 * y / (2.0*lambda) 
    - mu / (2.0*lambda) * pow(4.0*mu*lambda*y + mu2*pow(y, 2.0), 0.5);
  double z = R::runif(0.0, 1.0);
  double thresh =  mu/(mu+x);
  
  if(z<thresh){
    return x;
  }else{
    return mu2/x;
  }
}
