# libraries we (may) use in this experiment
library(Rcpp)
library(RcppArmadillo)
library(parallel)


library(devtools)
library(usethis)

library(BATTS)

library(foreach)
library(doParallel)

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

#data_train = read.csv("C:/Users/naway/Dropbox/Rcpp_experiments/data/Duke_Microbiome/sample_train.csv", header=F)
data_train = read.csv("C:/Users/user/Dropbox/Rcpp_experiments/data/Duke_Microbiome/sample_test.csv", header=F)

gen_names =  c("d", "dt", "icfm", "ltn", "mbgan", "otcfm")
#gen_names =  c("d")

# calculate quarantines
probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)
my_quantile = function(X){
  return(quantile(X, probs = probs_output_quantile))
}

# parameters
K_CV = 5
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

# parallelize!
n_cores = 3
cl =  makeCluster(n_cores)

registerDoParallel(cl)


# input data and modify them so that we can input them to our functions
foreach(index_gen=1:length(gen_names), .packages=c("balancePM")) %dopar% {

  set.seed(1)

  gen_name = gen_names[index_gen]

  #data_gen = read.csv(paste("C:/Users/naway/Dropbox/Rcpp_experiments/data/Duke_Microbiome/sample_",gen_name,".csv", sep=""), header=F)
  data_gen = read.csv(paste("C:/Users/user/Dropbox/Rcpp_experiments/data/Duke_Microbiome/sample_",gen_name,".csv", sep=""), header=F)

  data = as.matrix(rbind(data_train, data_gen))
  group_labels = c(rep(0, nrow(data_train)), rep(1, nrow(data_gen)))

  ########################################################################################################
  # method 1: proposed boosting (gradient based)
  set.seed(index_repeat)
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
  log_ratio_boosting_grad_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)

  ########################################################################################################
  # method 2: proposed boosting (Hellinger based)
  set.seed(index_repeat)
  result_estimation = boots(data = data,
                            group_labels = group_labels,
                            num_trees = num_trees_max,
                            K_CV = K_CV,
                            max_resol = max_depth,
                            learn_rate = learn_rate,
                            n_bins = n_bins,
                            use_gradient = F,
                            quiet = T
  )

  loss_hell = colMeans(result_estimation$loss_CV_store)
  log_ratio_boosting_hell_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)

  ########################################################################################################
  # method 3: proposed back-fitting (initialization: Hellinger based)
  set.seed(index_repeat)
  result_estimation = batts(data = data,
                            group_labels = group_labels,
                            num_trees = num_trees_Bayes,
                            # n_bins = n_bins,
                            size_burnin = size_burnin,
                            size_backfitting = size_backfitting,
                            output_BART_ensembles = FALSE,
                            quiet = T,
                            update_lambda = FALSE
  )

  # result of the BART
  log_ratio_BART_data = balance_to_log_ratio(result_estimation$balance_weight_BART_data)

  # obtain the quarantines
  log_ratio_BART_data_quantiles = apply(log_ratio_BART_data, 1, my_quantile)

  ########################################################################################################
  # output the result
  out = cbind(log_ratio_boosting_grad_data, log_ratio_boosting_hell_data, rowMeans(log_ratio_BART_data), t(log_ratio_BART_data_quantiles))
  out_name = paste("C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_microbiome/log_w_",gen_name,".csv", sep = "")
  # out_name = paste("C:/Users/user/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_microbiome/log_w_",gen_name,".csv", sep = "")

  write.table(out, file = out_name, sep = ",", row.names = FALSE, col.names = FALSE)

}

stopCluster(cl)
