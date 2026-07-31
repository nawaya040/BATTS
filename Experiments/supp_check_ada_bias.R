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

# log_ratio_grid_true = dnorm(grid_points, mu0, sqrt(sig20), log = TRUE) -
#   dnorm(grid_points, mu1, sqrt(sig21), log = TRUE)

# true_odds_ratio_grid = exp(log_ratio_grid_true) * n1 / n0
# true_class_prob_grid = true_odds_ratio_grid / (1 + true_odds_ratio_grid)

# the other settings
n_repeat = 50

K_CV = 0
num_trees_min = 10
num_trees_max = 100

learn_rate = 0.01
max_depth = 2

n_bins = 100

n_min_obs_per_node = 1

alpha_cutpoint = 1

setting_list = list()
setting_list[[1]] = c(500, 500)
setting_list[[2]] = c(300, 700)
setting_list[[3]] = c(200, 800)
setting_list[[4]] = c(100, 900)

grid_size = 100

n_cores = 5
cl =  makeCluster(n_cores)

registerDoParallel(cl)

foreach(index_repeat=1:n_repeat, .packages=c("ada", "BATTS")) %dopar% {

  for(index_setting in 1:length(setting_list)){

    n0 = setting_list[[index_setting]][1]
    n1 = setting_list[[index_setting]][2]

    group_labels = c(rep(0,n0), rep(1,n1))

    set.seed(index_repeat)

    # simulate the data
    data = matrix(c(rnorm(n0, mu0, sqrt(sig20)), rnorm(n1, mu1, sqrt(sig21))),ncol = 1)
    grid_points = seq(-2.5, 3.5, length.out = grid_size)

    # (1) Ada
    result_estimation = ada_log_weight(data,
                                       group_labels,
                                       grid_points,
                                       K_CV = K_CV,
                                       num_trees_max = num_trees_max,
                                       num_trees_min = num_trees_min,
                                       maxdepth = max_depth,
                                       learn_rate = learn_rate
    )

    user_name <- Sys.info()["user"]
    location_output_i = paste("C:/Users/", user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_1D/boosting",
                              sep="")

    name_output_i = paste("Adaboost", n0, n1, index_repeat, sep="_")

    saveRDS(result_estimation, paste(location_output_i, "/", name_output_i, ".rds", sep=""))

    # (2) Proposed boosting (gradient)
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

    name_output_i = paste("Proposed", n0, n1, index_repeat, sep="_")
    saveRDS(log_w_GS_grid, paste(location_output_i, "/", name_output_i, ".rds", sep=""))

  }

}

stopCluster(cl)

