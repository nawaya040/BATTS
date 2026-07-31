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

balance_to_log_ratio = function(x){
  return(log(x)*2)
}


# settings for simulation (a simple Gaussian model should be enough)
mu0 = 0
mu1 = 1
sig20 = 1^2
sig21 = 1.5^2

# the other settings
n_repeat = 10

K_CV = 5
num_trees_min = 10
num_trees_max = 1000
num_trees_Bayes = 200

learn_rate = 0.01
learn_rate_for_Bayes = 0.01

max_depth = 4
max_depth_for_Bayes = 0
n_min_obs_per_node = 1

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

setting_list = list()
setting_list[[1]] = c(500, 500)
setting_list[[2]] = c(500, 1000)
setting_list[[3]] = c(500, 1500)
setting_list[[4]] = c(500, 2000)
setting_list[[5]] = c(500, 2500)

n_cores = 5
cl =  makeCluster(n_cores)

registerDoParallel(cl)

foreach(index_repeat=1:n_repeat, .packages=c("ada", "BATTS")) %dopar% {

  for(index_setting in 1:length(setting_list)){

    n0 = setting_list[[index_setting]][1]
    n1 = setting_list[[index_setting]][2]

    set.seed(index_repeat)

    # simulate the data
    data = matrix(c(rnorm(n0, mu0, sqrt(sig20)), rnorm(n1, mu1, sqrt(sig21))),ncol = 1)

    # check the truth
    log_dens_ratio_data = dnorm(data, mu0, sqrt(sig20), log = TRUE) -
                            dnorm(data, mu1, sqrt(sig21), log = TRUE)

    group_labels = c(rep(0,n0), rep(1,n1))

    grid_points = seq(-2.5, 3.5, length.out = 100)

    log_ratio_grid_true = dnorm(grid_points, mu0, sqrt(sig20), log = TRUE) -
      dnorm(grid_points, mu1, sqrt(sig21), log = TRUE)

    # (1) boosting
    result_estimation = boots(data = data,
                              group_labels = group_labels,
                              num_trees = num_trees_max,
                              K_CV = K_CV,
                              max_resol = max_depth,
                              learn_rate = learn_rate,
                              n_bins = n_bins,
                              use_gradient = T,
                              quiet = T
    )

    # result of the boosting
    result_evaluation = eval_balance_weight(result_estimation, matrix(grid_points), is_Bayes = FALSE)

    log_w_GS_grid = balance_to_log_ratio(result_evaluation$balancing_weight_boosting)


    # (2) Adaboost
    result_estimation = ada_log_weight(data,
                                       group_labels,
                                       matrix(grid_points),
                                       K_CV = K_CV,
                                       num_trees_max = num_trees_max,
                                       num_trees_min = num_trees_min,
                                       maxdepth = max_depth,
                                       learn_rate = learn_rate
    )

    log_ratio_ada_grid = result_estimation$log_w_hat_ada_grid

    out = list("log_w_GS_grid" = log_w_GS_grid,
               "log_ratio_ada_grid" = log_ratio_ada_grid)

    user_name <- Sys.info()["user"]
    location_output_i = paste("C:/Users/", user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_1D/boosting",
                              sep="")

    name_output_i = paste("boosting_comparison", n0, n1, index_repeat, sep="_")

    saveRDS(out, paste(location_output_i, "/", name_output_i, ".rds", sep=""))

  }

}

stopCluster(cl)
