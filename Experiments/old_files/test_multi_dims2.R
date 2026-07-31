
#packages we use
library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(patchwork)
library(viridis)
library(parallel)
library(densratio)
library(mvtnorm)
library(xgboost)
library(rpart)
library(ada)

library(devtools)
library(usethis)

library(balancePM)
library(ada)
library(densratio)

library(foreach)
library(doParallel)

compute_sq_error = function(log_w_estimated, log_w_true_true){
  out = mean((log_w_estimated - log_w_true_true)^2)
  return(out)
}

# generate the data

n0 = 5000
n1 = 5000

d = 100
dim_dif_max = 2

group_labels = c(rep(0, n0), rep(1, n1))
n = n0 + n1
data = matrix(NA, nrow = n, ncol = d)

indices_0 = which(group_labels == 0)
indices_1 = which(group_labels == 1)

size_0 = length(indices_0)
size_1 = length(indices_1)

differences_log_dens = numeric(n)

mean0 = -0.5
sd0 = 1

mean1 = 0.5
sd1 = 1

k = 4

mu_vec0 = mu_vec1 = rep(mean0, d)
mu_vec1[dims_diff] = mean1

Sigma0 = Sigma1 = matrix(0, nrow = d, ncol = d)
diag(Sigma0) = sd0^2
diag(Sigma1) = sd0^2

for(j in 1:d){
  if(j <= dim_dif_max){
    data[indices_0,j] = rnorm(size_0, mean0, sd0)
    data[indices_1,j] = rnorm(size_1, mean1, sd1)
  }else{
    data[,j] = rnorm(n, mean0, sd0)
  }
}

# mix uniform components
unif_w = 0.2
range_x = c(mean0-k*sd0, mean1+k*sd1)

indices_unif = rbinom(n, size = 1, prob = unif_w)
n_unif = sum(indices_unif)

data[which(indices_unif==1), ] = matrix(runif(n_unif*d,range_x[1],range_x[2]),
                              nrow=n_unif, ncol=d)

d1 = 1
d2 = 2

plot(density(data[indices_0,d2]))
lines(density(data[indices_1,d2]), col = "red")

plot(data[indices_0,d1], data[indices_0,d2])
points(data[indices_1,d1], data[indices_1,d2], col = "red")

# compute the log-densities and the differences
mu_vec_0 = rep(mean0, d)
mu_vec_1 = c(rep(mean1, dim_dif_max), rep(mean0, d-dim_dif_max))

Sigma_0 = diag(rep(sd0^2, d))
Sigma_1 = diag(c(rep(sd1^2, dim_dif_max), rep(sd0^2, d-dim_dif_max)))

unif_dens = (range_x[2]-range_x[1])^(-d)
log_densities_0 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(data, mean = mu_vec_0, sigma = Sigma_0, log = F))
log_densities_1 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(data, mean = mu_vec_1, sigma = Sigma_1, log = F))

differences_log_dens = log_densities_0 - log_densities_1

plot(differences_log_dens)

# estimation

num_trees_max = 1000
K_CV = 5
max_depth = 4
learn_rate = 0.01

result_estimation = estimate_balancing_weight(data = data,
                                              group_labels = group_labels,
                                              num_trees = num_trees_max,
                                              K_CV = K_CV,
                                              max_resol = max_depth,
                                              learn_rate = learn_rate,
                                              use_gradient = F,
                                              size_burnin = 0,
                                              size_backfitting = 0,
                                              output_BART_ensembles = FALSE,
                                              quiet = FALSE
)

plot(colMeans(result_estimation$loss_CV_store))
plot(log(result_estimation$balance_weight_boosting_data)*2)


sqrt(compute_sq_error(differences_log_dens, log(result_estimation$balance_weight_boosting_data)*2))

plot(data[,1] + data[,2], log(result_estimation$balance_weight_boosting_data)*2)

plot(differences_log_dens - log(result_estimation$balance_weight_boosting_data)*2)
