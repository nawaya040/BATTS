
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

n0 = 1000
n1 = 1000

d = 100

group_labels = c(rep(0, n0), rep(1, n1))
n = n0 + n1
data = matrix(NA, nrow = n, ncol = d)

indices_0 = which(group_labels == 0)
indices_1 = which(group_labels == 1)

differences_log_dens = numeric(n)

mean0 = 0
sd0 = 1

mean1 = 1.0
sd1 = 1.5

means_vec = seq(mean0, mean1, length.out = d)
sd_vec = seq(sd0, sd1, length.out = d)

for(j in 1:d){
  data[indices_0,j] = rnorm(n0, mean0, sd0)
  data[indices_1,j] = rnorm(n0, means_vec[j], sd_vec[j])
}

d1 = 1
d2 = d

plot(density(data[indices_0,d2]))
lines(density(data[indices_1,d2]), col = "red")

plot(data[indices_0,d1], data[indices_0,d2])
points(data[indices_1,d1], data[indices_1,d2], col = "red")

differences_log_dens = numeric(n)

for(j in 1:d){
  differences_log_dens = differences_log_dens +
                                    dnorm(data[,j], mean0, sd0, log=TRUE) -
                                    dnorm(data[,j], means_vec[j], sd_vec[j], log=TRUE)
}

plot(differences_log_dens)

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


compute_sq_error(differences_log_dens, log(result_estimation$balance_weight_boosting_data)*2)
