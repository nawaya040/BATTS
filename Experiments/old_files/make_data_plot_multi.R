
# libraries we (may) use in this experiment
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

library(pracma)

source("./R/models_multi.R")
source("./R/utilities_for_experiment.R")

n0 = 2000
n1 = 2000

d = 100
unif_w = 0.5

transform = TRUE

n = n0 + n1

group_labels = c(rep(0, n0), rep(1, n1))
n = n0 + n1
data = matrix(NA, nrow = n, ncol = d)

indices_0 = which(group_labels == 0)
indices_1 = which(group_labels == 1)

size_0 = length(indices_0)
size_1 = length(indices_1)

differences_log_dens = numeric(n)

Q_full = randortho(d)

m = 4
U = Q_full[, 1:m]

dim(U)

mean_vec0 = c(0,0,0,0)
mean_vec1 = c(1.5,0,0,0)

sigma0 = diag(c(1,1,1,1)^2)
sigma1 = diag(c(1,1,1,1)^2)

data_u = matrix(NA, nrow = n, ncol = m)

sd_small = 0.1

mean_unif = c(0,0,0,0)
sd_unif = 3
sigma_unif = diag(rep(sd_unif, m)^2)

for(i in 1:n){

  prob_mix = runif(1)

  if(group_labels[i] == 0){
    if(prob_mix > unif_w){
      u = t(rmvnorm(1, mean_vec0, sigma0))
    }else{
      u = t(rmvnorm(1, mean_unif, sigma_unif))
    }
  }else{
    if(prob_mix > unif_w){
      u = t(rmvnorm(1, mean_vec1, sigma1))
    }else{
      u = t(rmvnorm(1, mean_unif, sigma_unif))
    }
  }
  data_u[i,] = u
}

#plot(data_u[which(group_labels == 0),1], data_u[which(group_labels == 0),2])
#points(data_u[which(group_labels == 1),1], data_u[which(group_labels == 1),2], col = "red")

data_before_transform = data_u %*% t(U) + matrix(rnorm(d*n,mean=0,sd=sd_small),nrow=n,ncol=d)

#plot(data[which(group_labels == 0),1], data[which(group_labels == 0),2])
#points(data[which(group_labels == 1),1], data[which(group_labels == 1),2], col = "red")

# compute the means and covariances of the d-dim distribution
mean0_vec_true = U %*% mean_vec0
mean1_vec_true = U %*% mean_vec1
mean_unif_vec_true = U %*% mean_unif

sigma0_true = U %*% sigma0 %*% t(U) + sd_small^2 * diag(d)
sigma1_true = U %*% sigma1 %*% t(U) + sd_small^2 * diag(d)
sigma_unif_true = U %*% sigma_unif %*% t(U) + sd_small^2 * diag(d)

# if necessary, transform to change the marginal distributions
if(transform){
  a_beta = 0.5
  b_beta = 2
  for(j in 1:d){
    data[,j] = qbeta(pnorm(data_before_transform[,j], mean_unif_vec_true[j], sqrt(sigma_unif_true[j,j])),a_beta,b_beta)
  }
}else{
  data = data_before_transform
}

par(mfrow = c(1,3))

plot(data_u[indices_0,1], data_u[indices_0,2], xlab = "u1", ylab = "u2", main = "Latent space")
points(data_u[indices_1,1], data_u[indices_1,2], col = "red")

plot(data_before_transform[indices_0,1], data_before_transform[indices_0,2], xlab = "x1", ylab = "x2", main = "Gaussian marginal")
points(data_before_transform[indices_1,1], data_before_transform[indices_1,2], col = "red")

plot(data[indices_0,1], data[indices_0,2], xlab = "x1", ylab = "x2", main = "Beta marginal")
points(data[indices_1,1], data[indices_1,2], col = "red")

par(mfrow = c(1,1))
