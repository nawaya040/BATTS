
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

compute_sq_error = function(log_w_estimated, log_w_true_true){
  out = mean((log_w_estimated - log_w_true_true)^2)
  return(out)
}

# generate the data

n0 = 1000
n1 = 1000
n = n0 + n1

d = 10

group_labels = c(rep(0, n0), rep(1, n1))
n = n0 + n1
data = matrix(NA, nrow = n, ncol = d)

indices_0 = which(group_labels == 0)
indices_1 = which(group_labels == 1)

size_0 = length(indices_0)
size_1 = length(indices_1)

differences_log_dens = numeric(n)

# parameter settings
library(pracma)
Q_full = randortho(d)

m = 4
U = Q_full[, 1:m]

dim(U)

mean_vec0 = c(0,0,0,0)
mean_vec1 = c(1,0,0,0)

sigma0 = diag(c(1,1,1,1)^2)
sigma1 = diag(c(1,1,1,1)^2)

data_eps = matrix(NA, nrow = n, ncol = m)

sd_small = 0.1

set.seed(10)

for(i in 1:n){
  if(group_labels[i] == 0){
    eps = t(rmvnorm(1, mean_vec0, sigma0))
    data_eps[i,] = eps
    data[i,] = U %*% eps + t(rmvnorm(1, sigma = sd_small^2 * diag(d)))
  }else{
    eps = t(rmvnorm(1, mean_vec1, sigma1))
    data_eps[i,] = eps
    data[i,] = U %*% eps + t(rmvnorm(1, sigma = sd_small^2 * diag(d)))
  }
}

# mix uniform components
d1 = 1
d2 = 2

plot(data[indices_0,d1], data[indices_0,d2], xlab = "x1", ylab = "x2")
points(data[indices_1,d1], data[indices_1,d2], col = "red")

d1 = 3
d2 = 5

plot(data[indices_0,d1], data[indices_0,d2], xlab = "x1", ylab = "x2")
points(data[indices_1,d1], data[indices_1,d2], col = "red")


plot(data_eps[indices_0,1], data_eps[indices_0,2])
points(data_eps[indices_1,1], data_eps[indices_1,2], col = "red")

data_pca = data %*% U

plot(data_pca[indices_0,1], data_pca[indices_0,2])
points(data_pca[indices_1,1], data_pca[indices_1,2], col = "red")

mean0_vec_true = U %*% mean_vec0
mean1_vec_true = U %*% mean_vec1

sigma0_true = U %*% sigma0 %*% t(U) + sd_small^2 * diag(d)
sigma1_true = U %*% sigma1 %*% t(U) + sd_small^2 * diag(d)

differences_log_dens = dmvnorm(data, mean0_vec_true, sigma0_true,log = TRUE) -
                          dmvnorm(data, mean1_vec_true, sigma1_true,log = TRUE)

plot(differences_log_dens)

#data_pca = matrix(NA, nrow = n, ncol = d)
#
#for(i in 1:n){
#  data_pca[i,] = t(t(U) %*% matrix(data[i,]))
#}
#
#plot(data_pca[indices_0,d1], data_pca[indices_0,d2])
#points(data_pca[indices_1,d1], data_pca[indices_1,d2], col = "red")

# estimation
num_trees_max = 1000
K_CV = 2
max_depth = 4
learn_rate = 0.01

result_estimation = estimate_balancing_weight(data = data,
                                              group_labels = group_labels,
                                              num_trees = num_trees_max,
                                              K_CV = K_CV,
                                              max_resol = max_depth,
                                              n_ratio_per_node = 0.1,
                                              learn_rate = learn_rate,
                                              use_gradient = F,
                                              size_burnin = 50,
                                              size_backfitting = 200,
                                              thin = 5,
                                              output_BART_ensembles = FALSE,
                                              quiet = FALSE
)

#saveRDS(result_estimation, "result_estimation2.rds")
#result_estimation = readRDS("result_estimation2.rds")

log_ratio_bosoting_data = log(result_estimation$balance_weight_boosting_data)*2
log_ratio_BART_data = rowMeans(log(result_estimation$balance_weight_BART_data)*2)

plot(log_ratio_bosoting_data, log_ratio_BART_data)
plot(differences_log_dens)

plot(colMeans(result_estimation$loss_CV_store))

sqrt(compute_sq_error(differences_log_dens, log_ratio_bosoting_data))
sqrt(compute_sq_error(differences_log_dens, log_ratio_BART_data))

plot(data_eps[,1], log(result_estimation$balance_weight_boosting_data)*2)
plot(data_eps[,2], log(result_estimation$balance_weight_boosting_data)*2)

plot(differences_log_dens - log(result_estimation$balance_weight_boosting_data)*2)


plot(result_estimation$balance_weight_BART_data[6,])

################################################################################################
# make a plot
log_ratio_boosting = log(result_estimation$balance_weight_boosting_data)*2
log_ratio_BART_mean = rowMeans(log(result_estimation$balance_weight_BART_data)*2)

probs_output_quantile = c(0.025, 0.975)

my_quantile = function(X){
  return(quantile(X, probs = probs_output_quantile))
}

log_ratio_BART_quantiles = apply(log(result_estimation$balance_weight_BART_data)*2, 1, my_quantile)

log_ratio_BART_lower = log_ratio_BART_quantiles[1,]
log_ratio_BART_upper = log_ratio_BART_quantiles[2,]

methods = c("Boosting", "Bayes (mean)", "Bayes (lower)", "Bayes (upper)")
n_methods = length(methods)

random_order = sample(1:n)
densities.df = data.frame(u1=rep(data_eps[random_order,1],n_methods),
                          u2=rep(data_eps[random_order,2],n_methods),
                          log_weight = c(log_ratio_boosting[random_order],
                                         log_ratio_BART_mean[random_order],
                                         log_ratio_BART_lower[random_order],
                                         log_ratio_BART_upper[random_order]),
                          method = factor(rep(methods, each=n), levels = methods)
)

min_true = min(differences_log_dens)
max_true = max(differences_log_dens)

min_all = min(densities.df$log_weight)
max_all = max(densities.df$log_weight)

scalle1 = 1

ggplot(densities.df, aes(x = u1, y = u2, color = log_weight)) +
  geom_point(size=3, alpha=0.75) +
  scale_color_gradientn(
    colours = c("darkblue",
                "blue",
                "deepskyblue",
                "cyan",
                "paleturquoise",
                "white",
                "white",
                "white",
                "lightyellow",
                "yellow",
                "orange",
                "red",
                "darkred"),
    values = scales::rescale(c(min_all,
                               min_all/2,
                               min_all/4,
                               min_all/8,
                               min_all/16,
                               min_all/10000,
                               0,
                               max_all/10000,
                               max_all/16,
                               max_all/8,
                               max_all/4,
                               max_all/2,
                               max_all)),  # 値の位置指定（0中心）
    limits = c(min_all, max_all),   # グラデーションはこの範囲に限定
    oob = scales::squish
  ) +
  facet_wrap(~method)+
  labs(color = "log(density)")
