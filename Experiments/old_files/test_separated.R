
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

#library(balancePM)
library(ada)
library(densratio)

library(foreach)
library(doParallel)

# what if the two distributions are completely separated?
n0 = 1000
n1 = 1000
n = n0 + n1

d=1
set.seed(1)

mean_1 = 5

X0 = matrix(rnorm(n0*d,mean=0), nrow=n0, ncol=d)
X1 = matrix(rnorm(n0*d,mean=mean_1), nrow=n1, ncol=d)

data = rbind(X0, X1)
group_labels = c(rep(0,n0), rep(1,n1))

# add some uniform components
unif_w = 0.5
k=4
range_min = 0 - 4
range_max = mean_1 + 4

indices_unif = rbinom(n, size = 1, prob = unif_w)
n_unif = sum(indices_unif)

for(j in 1:d){
  data[which(indices_unif==1),j] = runif(n_unif, min=range_min, max=range_max)
}

library(mvtnorm)
log_dens_true_0 = log((1-unif_w)*dmvnorm(data, mean=rep(0,d)) + unif_w/(range_max-range_min))
log_dens_true_1 = log((1-unif_w)*dmvnorm(data, mean=rep(mean_1,d)) + unif_w/(range_max-range_min))

log_w_true = log_dens_true_0 - log_dens_true_1

plot(data[1:n0,1], data[(n0+1):n,1])
plot(density(data[1:n0,1]), xlim = c(range_min, range_max))
lines(density(data[(n0+1):n,1]), xlim = c(range_min, range_max), col="red")
plot(data, log_w_true)

result_estimation = estimate_balancing_weight(data = data,
                                              group_labels = group_labels,
                                              num_trees = 1000,
                                              K_CV = 2,
                                              n_min_obs_per_node = 1,
                                              max_resol = 4,
                                              learn_rate = 0.01,
                                              use_gradient = F,
                                              size_burnin = 0,
                                              size_backfitting = 200,
                                              thin = 5,
                                              output_BART_ensembles = TRUE,
                                              quiet = F,
                                              update_lambda = TRUE
)

log_w_boosting_data = log(result_estimation$balance_weight_boosting_data) * 2
sd(log_w_boosting_data)
min_log_balance = min(log_w_boosting_data)
max_log_balance = max(log_w_boosting_data)
(max_log_balance - min_log_balance) / 9

quantiles_temp = quantile(log_w_boosting_data, probs = c(0.25, 0.75))
IQR = (quantiles_temp[2] - quantiles_temp[1]) / 1.35
IQR

log_w_BART_data = rowMeans(log(result_estimation$balance_weight_BART_data)) * 2
sd(log_w_BART_data)

sqrt(mean((log_w_boosting_data - log_w_true)^2))
sqrt(mean((log_w_BART_data - log_w_true)^2))

plot(log_w_true, log_w_boosting_data)
plot(log_w_true, log_w_BART_data)

plot(log(result_estimation$balance_weight_boosting_data))
plot(rowMeans(log(result_estimation$balance_weight_BART_data)))

plot(result_estimation$omega_store)

ind = 100
plot(log(result_estimation$balance_weight_BART_data[ind,]))
abline(h=log(result_estimation$balance_weight_boosting_data[ind]))
