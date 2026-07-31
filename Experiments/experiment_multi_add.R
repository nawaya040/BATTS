
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

output_all_thin = 5 # we output everything for a subset of the random seeds
#indices_to_check = seq(1, n_repeat-output_all_thin+1, by=output_all_thin)
indices_to_check = 1:n_repeat

probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)


K_CV = 5
num_trees_min = 10
num_trees_max = 1000
num_trees_Bayes = 200

learn_rate = 0.01
learn_rate_for_Bayes = 0.01
max_depth = 4
max_depth_for_Bayes = 0
n_min_obs_per_node = 1

n_bins = 32
alpha_cutpoint = 1

size_burnin = 2000
size_backfitting = 1000
thin = 1

lambda_0 = 5

d = 20
m = 4

# prep for cheeking the coverage rate
quant_probs = seq(0.005, 0.995, by = 0.005)
n_probs = length(quant_probs)
c_probs = 1 - quant_probs[1:(floor(n_probs/2))] * 2

# results should be put outside of the package directory!
user_name <- Sys.info()["user"]
location_output_root = paste(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results",sep=""))

# function to compute the squared error
compute_sq_error = function(log_w_estimated, log_w_true_true){
  out = mean((log_w_estimated - log_w_true_true)^2)
  return(out)
}

# settings for simulating data
# need to input (i) scenario name (ii) n0 (iii) n1 (iv)dims with differences
data_settings = list()　　
data_settings[[1]] = list("latent_location_shift", 5000, 5000, 0.2, FALSE)
data_settings[[2]] = list("latent_location_shift", 9000, 1000, 0.2, FALSE)
data_settings[[3]] = list("latent_dispersion", 5000, 5000, 0.2, FALSE)
data_settings[[4]] = list("latent_dispersion", 9000, 1000, 0.2, FALSE)

data_settings[[5]] = list("latent_location_shift", 5000, 5000, 0.2, TRUE)
data_settings[[6]] = list("latent_location_shift", 9000, 1000, 0.2, TRUE)
data_settings[[7]] = list("latent_dispersion", 5000, 5000, 0.2, TRUE)
data_settings[[8]] = list("latent_dispersion", 9000, 1000, 0.2, TRUE)

# update_only_BART0 = FALSE

size_spectral = 1000

n_cores = 10
cl =  makeCluster(n_cores)

registerDoParallel(cl)

# methods = c("Boosting(Gradient)", "Boosting(Hellinger)", "BART", "Real adaBoost", "KLIEP", "uLSIF")
#methods = c("Boosting(Gradient)", "Boosting(Hellinger)", "BART", "Real adaBoost", "uLSIF")
n_methods = 2

error_pooled_vec = numeric(n_methods)
error_group0_vec = numeric(n_methods)
error_group1_vec = numeric(n_methods)
#error_grid_vec = numeric(n_methods)


# generate the data and generate the grid points
# generate the data and generate the grid points
foreach(index_repeat=1:n_repeat, .packages=c("ada", "densityratio","mvtnorm","pracma")) %dopar% {

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
      location_output = paste(location_output_root, "experiments_multi", scenario, sep = "/")

      if(!transform){
        name_output = paste(scenario, "additional", n0, n1, lambda_0,index_repeat, sep = "_")
      }else{
        name_output = paste(scenario, "additional", n0, n1, "transformed", lambda_0, index_repeat, sep = "_")
      }
      ########################################################################################################
      # method 2-1: spectral method
      df0 = data.frame(data[indices_group0,])
      df1 = data.frame(data[indices_group1,])

      set.seed(index_repeat)
      dr_spectral = spectral(df_numerator = df1, # note: the smaller group is the numerator
                             df_denominator = df0,
                             parallel = FALSE, ncenters = size_spectral)

      result_predict = predict(dr_spectral, newdata = rbind(df0, df1), type = "densityratio")
      rep_value = max(1e-6, quantile(result_predict[,,1], 0.01))
      result_predict[which(result_predict[,,1] < 0)] = rep_value

      log_w_spectral_data = - log(result_predict[,,1])

      # save the memory
      if(index_repeat %% output_all_thin != 1){
        rm(dr_spectral)
      }

      error_pooled_vec[1] = compute_sq_error(log_w_spectral_data, log_w_true_data)
      error_group0_vec[1] = compute_sq_error(log_w_spectral_data[indices_group0], log_w_true_data[indices_group0])
      error_group1_vec[1] = compute_sq_error(log_w_spectral_data[indices_group1], log_w_true_data[indices_group1])

      ########################################################################################################
      # method 2-2: adaboost with the post calibration

      set.seed(index_repeat)
      result_estimation = ada_log_weight(data,
                                         group_labels,
                                         NULL,
                                         K_CV = K_CV,
                                         num_trees_max = num_trees_max,
                                         num_trees_min = num_trees_min,
                                         maxdepth = max_depth,
                                         learn_rate = learn_rate
      )

      logit_ada_data = result_estimation$log_w_hat_ada_data - log(n1 / n0)

      d0 = density(logit_ada_data[which(group_labels == 0)])
      d1 = density(logit_ada_data[which(group_labels == 1)])

      # evaluate the densities on logit_ada_data
      f0 = approx(x = d0$x, y = d0$y, xout = logit_ada_data, rule = 2)$y
      f1 = approx(x = d1$x, y = d1$y, xout = logit_ada_data, rule = 2)$y
      log_ratio_ada_data = log(f0 / f1)

      # save the memory
      if(index_repeat %% output_all_thin != 1){
        rm(result_estimation)
      }

      error_pooled_vec[2] = compute_sq_error(log_ratio_ada_data, log_w_true_data)
      error_group0_vec[2] = compute_sq_error(log_ratio_ada_data[indices_group0], log_w_true_data[indices_group0])
      error_group1_vec[2] = compute_sq_error(log_ratio_ada_data[indices_group1], log_w_true_data[indices_group1])


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
        "result_predict_spectral" = result_predict,
        "log_w_spectral_data" = log_w_spectral_data,
        "log_w_ada_data" = log_ratio_ada_data
      )

      saveRDS(out_details, paste(location_output, "/", name_output, "_details.rds", sep=""))
    }

    a = 1
    a

  }

}

stopCluster(cl)

#test1 = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_multi/latent_location_shift/latent_location_shift_900_100_transformed_5_1.rds")
#test2 = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_multi/latent_dispersion/latent_dispersion_900_100_transformed_5_3.rds")
#object.size(test)
#test_details = readRDS("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_2D/global_shift/global_shift_100_1000_1_details.rds")
#object.size(test_details)
