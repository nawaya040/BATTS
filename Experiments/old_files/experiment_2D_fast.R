
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

source("./R/models_2d.R")
source("./R/utilities_for_experiment.R")

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

# global experiment settings
n_repeat = 4

output_all_thin = 2 # we output everything for a subset of the random seeds

probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)

K_CV = 2
num_trees_min = 10
num_trees_max = 1000
num_trees_Bayes = 200

learn_rate = 0.01
learn_rate_for_Bayes = 0.01
max_depth = 4
max_depth_for_Bayes = 4
n_min_obs_per_node = 1

n_bins = 32
alpha_cutpoint = 1

size_burnin = 500
size_backfitting = 500
thin = 1

lambda_0 = 3

n_grid_per_dim = 100

# results should be put outside of the package directory!
user_name <- Sys.info()["user"]
location_output_root = paste(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results",sep=""))

# function to compute the squared error
compute_sq_error = function(log_w_estimated, log_w_true_true){
  out = mean((log_w_estimated - log_w_true_true)^2)
  return(out)
}

# settings for simulating data
# need to input (i) scenario name (ii) n0 (iii) n1
data_settings = list()
#data_settings[[1]] = list("global_shift", 1000, 5000)
#data_settings[[2]] = list("local_shift", 500, 500)
#data_settings[[3]] = list("local_dispersion", 500, 500)

#data_settings[[1]] = list("global_shift", 5000, 5000)
#data_settings[[2]] = list("global_shift", 9000, 1000)
#data_settings[[3]] = list("local_shift", 5000, 5000)
#data_settings[[4]] = list("local_shift", 9000, 1000)
data_settings[[1]] = list("global_shift", 5000, 5000)
data_settings[[2]] = list("global_shift", 9000, 1000)

n_cores = 1
cl =  makeCluster(n_cores)

registerDoParallel(cl)

methods = c("Boosting(Gradient)", "Boosting(Hellinger)", "BART")
#methods = c("Boosting(Gradient)", "Boosting(Hellinger)", "BART", "Real adaBoost", "uLSIF")
n_methods = length(methods)

#clarify which methods we want to use
methods_use = c(1,0,0)

error_pooled_vec = numeric(n_methods)
error_group0_vec = numeric(n_methods)
error_group1_vec = numeric(n_methods)

# generate the data and generate the grid points
foreach(index_repeat=1:n_repeat, .packages=c("balancePM","ada", "densratio")) %dopar% {

  for(index_settings in 1:length(data_settings)){

    # input the settings
    data_settings_current = data_settings[[index_settings]]

    scenario = data_settings_current[[1]]
    n0 = data_settings_current[[2]]
    n1 = data_settings_current[[3]]

    # check if the result already exists
    location_output = paste(location_output_root, "experiments_2D", scenario, sep = "/")
    name_output = paste(scenario, n0, n1, index_repeat, sep = "_")

    file_already_exists = file.exists(paste(location_output, "/", name_output, ".rds", sep=""))

    if(file_already_exists){
      prev_result = readRDS(paste(location_output, "/", name_output, ".rds", sep=""))

      error_pooled_vec = prev_result$error_pooled_vec
      error_group0_vec = prev_result$error_group0_vec
      error_group1_vec = prev_result$error_group1_vec

      if(index_repeat %% output_all_thin == 1){
        prev_result_details = readRDS(paste(location_output, "/", name_output, "_details.rds", sep=""))
      }
    }

    # simulate the data
    set.seed(index_repeat)
    out_data = simulation_2d(n0, n1, scenario, n_grid_per_dim)

    grid.points = out_data$grid_points
    n_eval = nrow(grid.points)

    data = out_data$data
    group_labels = out_data$group_labels

    indices_group0 = which(group_labels == 0)
    indices_group1 = which(group_labels == 1)

    log_w_true_data = out_data$true_log_w_obs
    log_w_true_grid = out_data$true_log_w_grid

    ########################################################################################################
    # method 1: proposed boosting (gradient based)
    if(methods_use[1] == 1){
      result_estimation = estimate_balancing_weight_boosting(data = data,
                                                             group_labels = group_labels,
                                                             num_trees = num_trees_max,
                                                             K_CV = K_CV,
                                                             max_resol = max_depth,
                                                             learn_rate = learn_rate,
                                                             n_bins = n_bins,
                                                             alpha_cutpoint = alpha_cutpoint,
                                                             n_min_obs_per_node = n_min_obs_per_node,
                                                             use_gradient = T,
                                                             quiet = T
      )

      # result of the boosting
      log_ratio_boosting_grad_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)

      # save the memory
      rm(result_estimation)

      # compute errors
      error_pooled_vec[1] = compute_sq_error(log_ratio_boosting_grad_data, log_w_true_data)
      error_group0_vec[1] = compute_sq_error(log_ratio_boosting_grad_data[indices_group0], log_w_true_data[indices_group0])
      error_group1_vec[1] = compute_sq_error(log_ratio_boosting_grad_data[indices_group1], log_w_true_data[indices_group1])

      # save the memory
      if(index_repeat %% output_all_thin != 1){
        rm(log_ratio_boosting_grad_data)
      }
    }else{
      if(index_repeat %% output_all_thin == 1){
        log_ratio_boosting_grad_data = prev_result_details$log_ratio_boosting_grad_data
      }
    }

    ########################################################################################################
    # method 2: proposed boosting (Hellinger based)
    if(methods_use[2] == 1){
      result_estimation = estimate_balancing_weight_boosting(data = data,
                                                             group_labels = group_labels,
                                                             num_trees = num_trees_max,
                                                             K_CV = K_CV,
                                                             max_resol = max_depth,
                                                             learn_rate = learn_rate,
                                                             n_bins = n_bins,
                                                             alpha_cutpoint = alpha_cutpoint,
                                                             n_min_obs_per_node = n_min_obs_per_node,
                                                             use_gradient = F,
                                                             quiet = T
      )

      loss_hell = colMeans(result_estimation$loss_CV_store)
      log_ratio_boosting_hell_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)

      # compute errors
      error_pooled_vec[2] = compute_sq_error(log_ratio_boosting_hell_data, log_w_true_data)
      error_group0_vec[2] = compute_sq_error(log_ratio_boosting_hell_data[indices_group0], log_w_true_data[indices_group0])
      error_group1_vec[2] = compute_sq_error(log_ratio_boosting_hell_data[indices_group1], log_w_true_data[indices_group1])

      # save the memory
      if(index_repeat %% output_all_thin != 1){
        rm(log_ratio_boosting_hell_data)
      }
    }else{
      if(index_repeat %% output_all_thin == 1){
        log_ratio_boosting_hell_data = prev_result_details$log_ratio_boosting_hell_data
      }
    }

    ########################################################################################################
    # method 3: proposed back-fitting (initialization: Hellinger based)
    if(methods_use[3] == 1){
      result_estimation = estimate_balancing_weight_Bayes(data = data,
                                                          group_labels = group_labels,
                                                          num_trees = num_trees_Bayes,
                                                          max_resol = max_depth_for_Bayes,
                                                          learn_rate = learn_rate_for_Bayes,
                                                          n_bins = n_bins,
                                                          alpha_cutpoint = alpha_cutpoint,
                                                          n_min_obs_per_node = n_min_obs_per_node,
                                                          use_gradient = F,
                                                          size_burnin = size_burnin,
                                                          size_backfitting = size_backfitting,
                                                          thin = thin,
                                                          prob_moves = c(1/3, 1/3, 1/3),
                                                          lambda_0 = lambda_0,
                                                          output_BART_ensembles = TRUE,
                                                          quiet = T,
                                                          update_lambda = FALSE
      )

      # result of the BART
      log_ratio_BART_data = balance_to_log_ratio(result_estimation$balance_weight_BART_data)
      log_ratio_BART_data_mean = rowMeans(log_ratio_BART_data)

      # save the memory
      rm(result_estimation)

      error_pooled_vec[3] = compute_sq_error(log_ratio_BART_data_mean, log_w_true_data)
      error_group0_vec[3] = compute_sq_error(log_ratio_BART_data_mean[indices_group0], log_w_true_data[indices_group0])
      error_group1_vec[3] = compute_sq_error(log_ratio_BART_data_mean[indices_group1], log_w_true_data[indices_group1])

      # save the memory
      if(index_repeat %% output_all_thin != 1){
        rm(log_ratio_BART_data_mean)
      }else{
        my_quantile = function(X){
          return(quantile(X, probs = probs_output_quantile))
        }

        log_ratio_BART_data_quantiles = apply(log_ratio_BART_data, 1, my_quantile)
      }

      rm(log_ratio_BART_data)

    }else{
      if(index_repeat %% output_all_thin == 1){
        log_ratio_BART_data_mean = prev_result_details$log_ratio_BART_data_mean
        log_ratio_BART_data_quantiles = prev_result_details$log_ratio_BART_data_quantiles
      }
    }

    ########################################################################################################
    # output results
    out = list("error_pooled_vec" = error_pooled_vec,
               "error_group0_vec" = error_group0_vec,
               "error_group1_vec" = error_group1_vec
    )

    saveRDS(out, paste(location_output, "/", name_output, ".rds", sep=""))

    # we also need to the estimated functions for drawing figures
    if(index_repeat %% output_all_thin == 1){
      out_details = list(
        "log_ratio_boosting_grad_data" = log_ratio_boosting_grad_data,
        "log_ratio_boosting_hell_data" = log_ratio_boosting_hell_data,
        "log_ratio_BART_data_mean" = log_ratio_BART_data_mean,
        "log_ratio_BART_data_quantiles" = log_ratio_BART_data_quantiles
      )

      saveRDS(out_details, paste(location_output, "/", name_output, "_details.rds", sep=""))

    }

    a = 1
    a

  }

}

stopCluster(cl)
#test = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_2D/global_shift/global_shift_1000_1000_1.rds")
#object.size(test)
#test_details = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_2D/global_shift/global_shift_100_1000_1_details.rds")
#object.size(test_details)
