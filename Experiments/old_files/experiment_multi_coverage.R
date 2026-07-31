
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

library(BATTS)
library(ada)
library(densratio)

library(foreach)
library(doParallel)

library(pracma)

source("./R/models_multi.R")
source("./R/utilities_for_experiment.R")

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

# global experiment settings
n_repeat = 50

# K_CV = 2
# num_trees_min = 10
# num_trees_max = 1000
num_trees_Bayes = 200

# learn_rate = 0.01
# learn_rate_for_Bayes = 0.01
# max_depth = 4
# max_depth_for_Bayes = 4
# n_min_obs_per_node = 1

 n_bins = 32
# alpha_cutpoint = 1

size_burnin = 2000
size_backfitting = 1000
# thin = 1

lambda_0 = 5

d = 20
m = 4

# results should be put outside of the package directory!
user_name <- Sys.info()["user"]
location_output_root = paste(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results",sep=""))

# function to compute the squared error
# compute_sq_error = function(log_w_estimated, log_w_true_true){
#   out = mean((log_w_estimated - log_w_true_true)^2)
#   return(out)
# }

# settings for simulating data
# need to input (i) scenario name (ii) n0 (iii) n1 (iv)dims with differences
data_settings = list()
# data_settings[[1]] = list("latent_location_shift", 1000, 1000, 0.2, FALSE)

scale = 1

data_settings[[1]] = list("latent_location_shift", 5000*scale, 5000*scale, 0.2, FALSE)
data_settings[[2]] = list("latent_location_shift", 9000*scale, 1000*scale, 0.2, FALSE)
data_settings[[3]] = list("latent_dispersion", 5000*scale, 5000*scale, 0.2, FALSE)
data_settings[[4]] = list("latent_dispersion", 9000*scale, 1000*scale, 0.2, FALSE)

data_settings[[5]] = list("latent_location_shift", 5000*scale, 5000*scale, 0.2, TRUE)
data_settings[[6]] = list("latent_location_shift", 9000*scale, 1000*scale, 0.2, TRUE)
data_settings[[7]] = list("latent_dispersion", 5000*scale, 5000*scale, 0.2, TRUE)
data_settings[[8]] = list("latent_dispersion", 9000*scale, 1000*scale, 0.2, TRUE)

scale = 5

data_settings[[9]] = list("latent_location_shift", 5000*scale, 5000*scale, 0.2, FALSE)
data_settings[[10]] = list("latent_location_shift", 9000*scale, 1000*scale, 0.2, FALSE)
data_settings[[11]] = list("latent_dispersion", 5000*scale, 5000*scale, 0.2, FALSE)
data_settings[[12]] = list("latent_dispersion", 9000*scale, 1000*scale, 0.2, FALSE)

data_settings[[13]] = list("latent_location_shift", 5000*scale, 5000*scale, 0.2, TRUE)
data_settings[[14]] = list("latent_location_shift", 9000*scale, 1000*scale, 0.2, TRUE)
data_settings[[15]] = list("latent_dispersion", 5000*scale, 5000*scale, 0.2, TRUE)
data_settings[[16]] = list("latent_dispersion", 9000*scale, 1000*scale, 0.2, TRUE)

scale = 10

data_settings[[17]] = list("latent_location_shift", 5000*scale, 5000*scale, 0.2, FALSE)
data_settings[[18]] = list("latent_location_shift", 9000*scale, 1000*scale, 0.2, FALSE)
data_settings[[19]] = list("latent_dispersion", 5000*scale, 5000*scale, 0.2, FALSE)
data_settings[[20]] = list("latent_dispersion", 9000*scale, 1000*scale, 0.2, FALSE)

data_settings[[21]] = list("latent_location_shift", 5000*scale, 5000*scale, 0.2, TRUE)
data_settings[[22]] = list("latent_location_shift", 9000*scale, 1000*scale, 0.2, TRUE)
data_settings[[23]] = list("latent_dispersion", 5000*scale, 5000*scale, 0.2, TRUE)
data_settings[[24]] = list("latent_dispersion", 9000*scale, 1000*scale, 0.2, TRUE)

n_cores = 5
cl =  makeCluster(n_cores)

registerDoParallel(cl)

#methods = c("Boosting(Gradient)", "Boosting(Hellinger)", "BART")
#methods = c("Boosting(Gradient)", "Boosting(Hellinger)", "BART", "Real adaBoost", "uLSIF")
#n_methods = length(methods)

# error_pooled_vec = numeric(n_methods)
# error_group0_vec = numeric(n_methods)
# error_group1_vec = numeric(n_methods)
#error_grid_vec = numeric(n_methods)

# generate the data and generate the grid points
foreach(index_repeat=1:n_repeat, .packages=c("BATTS","mvtnorm","pracma")) %dopar% {

  for(index_settings in 1:length(data_settings)){

      data_settings_current = data_settings[[index_settings]]

      scenario = data_settings_current[[1]]
      n0 = data_settings_current[[2]]
      n1 = data_settings_current[[3]]
      unif_w = data_settings_current[[4]]
      transform = data_settings_current[[5]]
      #dim_dif_max = data_settings_current[[4]]

      set.seed(index_repeat)
      #out_data = simulation_multi_latent(n0, n1, d, m, scenario, unif_w)
      out_data = simulation_multi_latent(n0, n1, d, scenario, unif_w, transform)

      data = out_data$data
      group_labels = out_data$group_labels

      indices_group0 = which(group_labels == 0)
      indices_group1 = which(group_labels == 1)

      log_w_true_data = out_data$true_log_w_obs

      ########################################################################################################
      # # method 1: proposed boosting (gradient based)
      # result_estimation = estimate_balancing_weight_boosting(data = data,
      #                                                        group_labels = group_labels,
      #                                                        num_trees = num_trees_max,
      #                                                        K_CV = K_CV,
      #                                                        max_resol = max_depth,
      #                                                        learn_rate = learn_rate,
      #                                                        n_bins = n_bins,
      #                                                        alpha_cutpoint = alpha_cutpoint,
      #                                                        n_min_obs_per_node = n_min_obs_per_node,
      #                                                        use_gradient = T,
      #                                                        quiet = T
      # )
      #
      # # result of the boosting
      # log_ratio_boosting_grad_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)
      #
      # # save the memory
      # rm(result_estimation)
      #
      # # compute errors
      # error_pooled_vec[1] = compute_sq_error(log_ratio_boosting_grad_data, log_w_true_data)
      # error_group0_vec[1] = compute_sq_error(log_ratio_boosting_grad_data[indices_group0], log_w_true_data[indices_group0])
      # error_group1_vec[1] = compute_sq_error(log_ratio_boosting_grad_data[indices_group1], log_w_true_data[indices_group1])
      #
      # # save the memory
      # if(index_repeat %% output_all_thin != 1){
      #   rm(log_ratio_boosting_grad_data)
      # }
      #
      # ########################################################################################################
      # # method 2: proposed boosting (Hellinger based)
      # result_estimation = estimate_balancing_weight_boosting(data = data,
      #                                                        group_labels = group_labels,
      #                                                        num_trees = num_trees_max,
      #                                                        K_CV = K_CV,
      #                                                        max_resol = max_depth,
      #                                                        learn_rate = learn_rate,
      #                                                        n_bins = n_bins,
      #                                                        alpha_cutpoint = alpha_cutpoint,
      #                                                        n_min_obs_per_node = n_min_obs_per_node,
      #                                                        use_gradient = F,
      #                                                        quiet = T
      # )
      #
      # loss_hell = colMeans(result_estimation$loss_CV_store)
      # log_ratio_boosting_hell_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)
      #
      # # compute errors
      # error_pooled_vec[2] = compute_sq_error(log_ratio_boosting_hell_data, log_w_true_data)
      # error_group0_vec[2] = compute_sq_error(log_ratio_boosting_hell_data[indices_group0], log_w_true_data[indices_group0])
      # error_group1_vec[2] = compute_sq_error(log_ratio_boosting_hell_data[indices_group1], log_w_true_data[indices_group1])
      #
      # # save the memory
      # if(index_repeat %% output_all_thin != 1){
      #   rm(log_ratio_boosting_hell_data)
      # }

      ########################################################################################################
      # method 3: proposed back-fitting (initialization: Hellinger based)
      result_estimation = batts(data = data,
                                group_labels = group_labels,
                                num_trees = num_trees_Bayes,
                                n_bins = n_bins,
                                size_burnin = size_burnin,
                                size_backfitting = size_backfitting,
                                output_BART_ensembles = FALSE,
                                lambda_0 = lambda_0,
                                quiet = F,
                                update_lambda = FALSE
      )

      # result of the BART
      log_ratio_BART_data = balance_to_log_ratio(result_estimation$balance_weight_BART_data)
      #log_ratio_BART_data_mean = rowMeans(log_ratio_BART_data)

      # save the memory
      rm(result_estimation)

      # compute the quantiles
      log_ratio_BART_data_quantiles = apply(log_ratio_BART_data, 1, quantile, probs = c(0.025, 0.05, 0.95, 0.975))

      covered_90 = sapply(1:nrow(data), function(i){
        (log_ratio_BART_data_quantiles["5%",i] < log_w_true_data[i]) * (log_ratio_BART_data_quantiles["95%",i] > log_w_true_data[i])
      })

      names(covered_90) = NULL

      covered_95 = sapply(1:nrow(data), function(i){
        (log_ratio_BART_data_quantiles["2.5%",i] < log_w_true_data[i]) * (log_ratio_BART_data_quantiles["97.5%",i] > log_w_true_data[i])
      })

      names(covered_95) = NULL

    ########################################################################################################
    # output results
    out = list("log_w_true_data" = log_w_true_data,
               "covered_90" = covered_90,
               "covered_95" = covered_95
    )

    location_output = paste(location_output_root, "experiments_multi", scenario, sep = "/")

    if(transform){
      name_output = paste(scenario, n0, n1, lambda_0, index_repeat, "coverage_transformed", sep = "_")
    }else{
      name_output = paste(scenario, n0, n1, lambda_0, index_repeat, "coverage", sep = "_")
    }


    saveRDS(out, paste(location_output, "/", name_output, ".rds", sep=""))

    a = 1
    a

  }

}

stopCluster(cl)

#test1 = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_multi/latent_location_shift/latent_location_shift_4000_400_1.rds")
#test2 = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_multi/latent_dispersion/latent_dispersion_2000_500_4_38.rds")
#object.size(test)
#test_details = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_2D/global_shift/global_shift_100_1000_1_details.rds")
#object.size(test_details)
