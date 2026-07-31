library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(tidyr)
library(dplyr)
library(patchwork)
library(viridis)
library(parallel)
library(densratio)
library(mvtnorm)
# library(xgboost)
library(rpart)
library(ada)

library(devtools)
library(usethis)

library(BATTS)
library(ada)
library(densratio)

library(foreach)
library(doParallel)

library(patchwork)

source("./R/models_2d.R")
source("./R/models_multi.R")
source("./R/utilities_for_experiment.R")

windowsFonts(Helvetica = windowsFont("Helvetica"),
             Arial = windowsFont("Arial"))

# settings for simulation (a simple Gaussian model should be enough)
mu0 = 0
mu1 = 1
sig20 = 1^2
sig21 = 1.5^2

BD_true = 1/4 * (mu0 - mu1)^2 / (sig20 + sig21) +
  1/2 * log((sig20 + sig21) / (2 * sqrt(sig20 * sig21)))
BC_true = exp(-BD_true)


# the other settings
n_repeat = 50

num_trees_Bayes = 200

learn_rate_for_Bayes = 0.01
max_depth_for_Bayes = 0
n_min_obs_per_node = 1

n_bins = 1000
alpha_cutpoint = 1

size_burnin = 1000
size_backfitting = 2000
thin = 1

lambda_0 = 5

n_grid_per_dim = 100

quant_probs = seq(0.005, 0.995, by = 0.005)
n_probs = length(quant_probs)
c_probs = 1 - quant_probs[1:(floor(n_probs/2))] * 2

n_c_probs = length(c_probs)

scale = 5

setting_list = list()
setting_list[[1]] = c(500, 500)
setting_list[[2]] = c(100, 900)
setting_list[[3]] = c(500, 500) * scale
setting_list[[4]] = c(100, 900) * scale

n_cores = 5
cl =  makeCluster(n_cores)

registerDoParallel(cl)

foreach(index_repeat=1:n_repeat, .packages=c("BATTS")) %dopar% {

  cov_rate_store = matrix(NA, nrow=length(setting_list), ncol=n_c_probs)

  output_details = (index_repeat == 1)

  for(index_setting in 1:length(setting_list)){

    n0 = setting_list[[index_setting]][1]
    n1 = setting_list[[index_setting]][2]

    set.seed(index_repeat)

    # simulate the data
    data = matrix(c(rnorm(n0, mu0, sqrt(sig20)), rnorm(n1, mu1, sqrt(sig21))),ncol = 1)

    # check the truth
    log_dens_ratio_data = dnorm(data, mu0, sqrt(sig20), log = TRUE) -
                            dnorm(data, mu1, sqrt(sig21), log = TRUE)

    if(output_details){
      grid_points = seq(-2.5, 3.5, length.out = 100)
    }

    result_estimation = batts(data = data,
                              group_labels = c(rep(0,n0), rep(1,n1)),
                              num_trees = num_trees_Bayes,
                              # n_bins = n_bins,
                              size_burnin = size_burnin,
                              size_backfitting = size_backfitting,
                              output_BART_ensembles = output_details,
                              lambda_0 = lambda_0,
                              quiet = T,
                              update_lambda = FALSE
    )

    CIs =  apply(log(result_estimation$balance_weight_BART_data) * 2,
                 1, quantile, probs = quant_probs)

    if(output_details){
      tau_inv_result = result_estimation$omega_store^(-1)

      eval_BAT = eval_balance_weight(list_result = result_estimation,
                                     eval_points = matrix(grid_points),
                                     is_Bayes = T)

      log_ratio_grid_post_mean = rowMeans(log(eval_BAT$balancing_weight_BART) * 2)
      log_ratio_grid_true = dnorm(grid_points, mu0, sqrt(sig20), log = TRUE) -
        dnorm(grid_points, mu1, sqrt(sig21), log = TRUE)

      CIs_grid = apply(log(eval_BAT$balancing_weight_BART) * 2,
                   1, quantile, probs = quant_probs)

      out = list("log_ratio_grid_post_mean" = log_ratio_grid_post_mean,
                 "log_ratio_grid_true" = log_ratio_grid_true,
                 "tau_inv_result" = tau_inv_result,
                 "CIs" = CIs_grid)




    }


    # check coverage rates
    cov_rates = numeric(n_c_probs)

    for(index_c in 1:n_c_probs){
      cov_rates[index_c] = mean((CIs[index_c,] < log_dens_ratio_data) * (log_dens_ratio_data < CIs[n_probs - index_c,]))
    }

    cov_rate_store[index_setting,] = cov_rates

    user_name <- Sys.info()["user"]
    location_output_i = paste("C:/Users/", user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_1D/illustration",
                            sep="")

    name_output_i = paste("illustration", n0, n1, sep="_")

    saveRDS(out, paste(location_output_i, "/", name_output_i, ".rds", sep=""))
  }


  user_name <- Sys.info()["user"]
  location_output = paste("C:/Users/", user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_1D/coverage",
                          sep="")
  name_output = paste("coverage", index_repeat, sep="_")

  saveRDS(cov_rate_store, paste(location_output, "/", name_output, ".rds", sep=""))

  a = 1
  a

}

stopCluster(cl)
