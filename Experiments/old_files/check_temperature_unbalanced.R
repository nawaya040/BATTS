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

library(BATTS)
library(ada)
library(densratio)

library(foreach)
library(doParallel)

source("./R/models_2d.R")
source("./R/models_multi.R")
source("./R/utilities_for_experiment.R")

# tuning parameters for the back-fitting
n_repeat = 10

output_all_thin = 5 # we output everything for a subset of the random seeds

probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)

num_trees_Bayes = 200

learn_rate_for_Bayes = 0.01
max_depth_for_Bayes = 0
n_min_obs_per_node = 1

n_bins = 1000
alpha_cutpoint = 1

size_burnin = 500
size_backfitting = 500
thin = 1

lambda_0 = 5

n_grid_per_dim = 100

data_settings = list()

#data_settings[[1]] = list("global_shift", 9000, 1000)
#data_settings[[2]] = list("local_shift", 9000, 1000)
#data_settings[[3]] = list("local_dispersion", 9000, 1000)

#data_settings[[1]] = list("latent_location_shift", 9000, 1000, 0.2, FALSE)
#data_settings[[2]] = list("latent_dispersion", 9000, 1000, 0.2, FALSE)

n0 = 100
n1 = 500

group_labels = c(rep(0,n0), rep(1,n1))

indices_0 = which(group_labels == 0)
indices_1 = which(group_labels == 1)

omega_result_store = matrix(NA, nrow = 3, ncol = size_backfitting / thin)
low_ratio_result_store = matrix(NA, nrow = 3, ncol = n0 + n1)
data_store = matrix(NA, nrow = 3, ncol = n0 + n1)

lower_store = matrix(NA, nrow = 3, ncol = n0 + n1)
upper_store = matrix(NA, nrow = 3, ncol = n0 + n1)

true_store = matrix(NA, nrow = 3, ncol = n0 + n1)


for(index_settings in 1:3){

  #data_settings_current = data_settings[[index_settings]]
  #scenario = data_settings_current[[1]]
  #n0 = data_settings_current[[2]]
  #n1 = data_settings_current[[3]]

  set.seed(1)
  #out_data = simulation_2d(n0, n1, scenario, 100)
  #out_data = simulation_multi_latent(n0, n1, 20, scenario, 0.2, FALSE)

  #data = out_data$data
  #group_labels = out_data$group_labels

  data = matrix(c(rnorm(n0), rnorm(n1)+index_settings), ncol = 1)

  ########################################################################################################
  # method 3: proposed back-fitting (initialization: Hellinger based)
  result_estimation = batts(data = data,
                            group_labels = group_labels,
                            num_trees = num_trees_Bayes,
                            # n_bins = n_bins,
                            size_burnin = size_burnin,
                            size_backfitting = size_backfitting,
                            output_BART_ensembles = FALSE,
                            lambda_0 = lambda_0,
                            quiet = T,
                            update_lambda = FALSE
  )

  omega_result_store[index_settings,] = result_estimation$omega_store
  data_store[index_settings,] = data
  low_ratio_result_store[index_settings,] = rowMeans(log(result_estimation$balance_weight_BART_data) * 2)
  CI_temp =  apply(log(result_estimation$balance_weight_BART_data) * 2,
                   1, quantile, probs = c(0.025, 0.975))

  lower_store[index_settings,] = CI_temp[1,]
  upper_store[index_settings,] = CI_temp[2,]

  true_store[index_settings, ] = dnorm(data, log=TRUE) - dnorm(data,index_settings, log=TRUE)
}

random_order = sample(1:(n0+n1))
point_colors = c(rep("black", n0), rep("red", n1))[random_order]
point_shape = c(rep(1, n0), rep(2, n1))[random_order]
location_y = c(rep(0.00, n0), rep(-0.00, n1))[random_order]

par(mfcol = c(2,3), mar = c(5.1, 4.1, 4.1, 2.1))
for(index_settings in 1:3){
  true_hell_dist_sq = 1 - exp( - index_settings^2 / 8)

  plot(data_store[index_settings,random_order], location_y,
       col = point_colors,
       pch = point_shape,
       xlim = range(data_store[index_settings,]),
       ylim = range(rbind(lower_store, upper_store)),
       xlab = "x", ylab = "log(density ratio)",
       main = paste("Shift = ",index_settings," x SD", sep = ""),
       cex.lab = 1.0
       )

  sort.result = sort(data_store[index_settings,], index.return=TRUE)

  lines(sort.result$x, low_ratio_result_store[index_settings,sort.result$ix], lwd = 2)
  lines(sort.result$x, lower_store[index_settings,sort.result$ix], lwd = 2, col = "blue")
  lines(sort.result$x, upper_store[index_settings,sort.result$ix], lwd = 2, col = "blue")

  lines(sort.result$x, true_store[index_settings,sort.result$ix], lwd = 2, col = "gray", lty = 2)

  plot(density(1 - omega_result_store[index_settings,]^(-1)),
       main = true_hell_dist_sq,
       xlab = expression(tau),
       ylab = "Density",
       # xlim = c(0.4, 2),
       lwd = 1.5,
       cex.lab = 1.0
       )
}
par(mfrow = c(1,1),mar = c(5.1, 4.1, 4.1, 2.1))
