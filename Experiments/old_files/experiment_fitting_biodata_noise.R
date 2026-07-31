# libraries we (may) use in this experiment
library(Rcpp)
library(RcppArmadillo)
library(parallel)

library(ada)
library(densratio)

library(devtools)
library(usethis)

library(BATTS)

library(foreach)
library(doParallel)

source("./R/utilities_for_experiment.R")

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

user_name <- Sys.info()["user"]
file_name = paste(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/data/Duke_Microbiome/sample_train.csv",sep=""))

data_raw0 = read.csv(file_name, header=F)

gen_names =  c("mbgan")
#gen_names =  c("icfm")

# calculate quarantines
probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)
my_quantile = function(X){
  return(quantile(X, probs = probs_output_quantile))
}

# parameters
K_CV = 2
num_trees_max = 1000
num_trees_min = 200
num_trees_Bayes = 200

learn_rate = 0.01
learn_rate_for_Bayes = 0.01
max_depth = 4
max_depth_for_Bayes = 0
n_min_obs_per_node = 1

n_bins = 32
# alpha_cutpoint = 1

size_burnin = 2000
size_backfitting = 1000
thin = 1

lambda_0 = 5

# parallelize!
# n_cores = 3
# cl =  makeCluster(n_cores)
#
# registerDoParallel(cl)


# input data and modify them so that we can input them to our functions

index_gen = 1

#foreach(index_gen=1:length(gen_names), .packages=c("balancePM")) %dopar% {

  index_seed = 1

  gen_name = gen_names[index_gen]

  file_name_gen = paste(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/data/Duke_Microbiome/sample_",gen_name,".csv",sep=""))
  data_gen = read.csv(file_name_gen, header=F)

  set.seed(index_seed)

  # for simplicity, make both sample sizes equal to 1000
  n = 1000
  n_sub = 500
  data_raw = data_raw0[sort(sample(1:nrow(data_raw0), n)),]

  data_combined = rbind(data_gen, data_raw)

  # randomly split the data into training data and test data
  # since the data show poor overlapping, we add "noise" by simulating from the combined sample
  indices_train = sort(sample(1:n, size = n_sub))
  indices_test = (1:n)[-indices_train]

  indices_noise_train = sort(sample(1:(2*n), size = n_sub))
  indices_noise_test = (1:(2*n))[-indices_noise_train]

  data_train = as.matrix(rbind(data_gen[indices_train,],
                        data_combined[indices_noise_train,],
                        data_raw[indices_train,],
                        data_combined[indices_noise_train,]))

  group_labels_train = c(rep(0, n), rep(1, n))

  data_eval = as.matrix(rbind(data_gen[indices_test,],
                              data_combined[indices_noise_test,],
                              data_raw[indices_test,],
                              data_combined[indices_noise_test,]))

  n_gen_eval = n_gen - n_gen_train
  n_eval = nrow(data_eval)

  # define the function to compute the squared error
  sq_error = function(dens_ratios){
    1/2 * mean(dens_ratios[(n_gen_eval+1):n_eval]^2) - mean(dens_ratios[1:n_gen_eval])
  }

  # temporal: simplify the data
  # data_train = data_train[,c(1:10)]
  # data_eval = data_eval[,c(1:10)]

  plot(data_train[,1], data_train[,2])
  points(data_eval[,1], data_eval[,2], col = "red")

  ########################################################################################################
  # method 1: real adaboost

  result_estimation = ada_log_weight(data_train,
                                     group_labels_train,
                                     data_eval,
                                     K_CV = K_CV,
                                     num_trees_max = num_trees_max,
                                     num_trees_min = num_trees_min,
                                     maxdepth = max_depth,
                                     learn_rate = learn_rate
  )

  dens_ratio_ada_eval = exp(result_estimation$log_w_hat_ada_grid)

  # save the memory
  rm(result_estimation)

  sq_error(dens_ratio_ada_eval)
  1/2 * mean(dens_ratio_ada_eval[(n_gen_eval+1):n_eval]^2) - mean(dens_ratio_ada_eval[1:n_gen_eval])

  ########################################################################################################
  # method 2: densratio methods

  X0 = data_train[which(group_labels_train==0),]
  X1 = data_train[which(group_labels_train==1),]

  result_estimation = KLIEP(X0, X1, verbose = T)
  dens_ratio_KLIEP_eval = result_estimation$compute_density_ratio(data_eval)

  # save the memory
  rm(result_estimation)

  # compute the error
  sq_error(dens_ratio_KLIEP_eval)

  result_estimation = uLSIF(X0, X1, verbose = FALSE)
  dens_ratio_uLSIF_eval  = result_estimation$compute_density_ratio(data_eval)

  # save the memory
  rm(result_estimation)

  # compute the error
  sq_error(dens_ratio_uLSIF_eval)

  ########################################################################################################
  # method 3: proposed boosting (gradient based)
  result_estimation = boots(data = data_train,
                            group_labels = group_labels_train,
                            num_trees = num_trees_max,
                            K_CV = K_CV,
                            max_resol = max_depth,
                            learn_rate = learn_rate,
                            n_bins = n_bins,
                            use_gradient = T,
                            quiet = F
  )

  # result of the boosting
  dens_ratio_boosting_grad_eval = result_estimation$balance_weight_boosting_data^2

  # save the memory
  rm(result_estimation)

  # compute errors
  print(sq_error(dens_ratio_boosting_grad_eval))


  ########################################################################################################
  # method 5: proposed back-fitting (initialization: Hellinger based)
  result_estimation = batts(data = data_train,
                            group_labels = group_labels_train,
                            num_trees = num_trees_Bayes,
                            n_bins = n_bins,
                            size_burnin = size_burnin,
                            size_backfitting = size_backfitting,
                            output_BART_ensembles = TRUE,
                            lambda_0 = lambda_0,
                            quiet = F,
                            update_lambda = FALSE,
                            margin_scale = -1
  )

  result_evaluation = eval_balance_weight(result_estimation, data_eval, is_Bayes = TRUE)

  # save the memory
  rm(result_estimation)

  # obtain the posterior means for the density ratios (not log!)
  dens_ratio_BART_data_mean = rowMeans(result_evaluation$balancing_weight_BART^2)

  # compute the error
  sq_error(dens_ratio_BART_data_mean)

  ########################################################################################################
  # output the result
  out = cbind(log_ratio_boosting_grad_data, log_ratio_boosting_hell_data, rowMeans(log_ratio_BART_data), t(log_ratio_BART_data_quantiles))
  #out_name = paste("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_microbiome/log_w_",gen_name,".csv", sep = "")
  out_name = paste("C:/Users/user/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_microbiome/log_w_",gen_name,".csv", sep = "")

  write.table(out, file = out_name, sep = ",", row.names = FALSE, col.names = FALSE)

#}

# stopCluster(cl)
