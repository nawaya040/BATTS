
# libraries we (may) use in this experiment
library(mvtnorm)
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
data_settings[[1]] = list("latent_location_shift", 500, 500, 0.2, TRUE)
# data_settings[[2]] = list("latent_location_shift", 900, 100, 0.2, TRUE)
# data_settings[[3]] = list("latent_dispersion", 500, 500, 0.2, TRUE)
# data_settings[[4]] = list("latent_dispersion", 900, 100, 0.2, TRUE)


index_repeat = 1

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
  out_data = simulation_multi_latent(n0, n1, d, scenario, unif_w, transform,
                                     a_beta = 0.5, b_beta = 10)

  data = out_data$data
  group_labels = out_data$group_labels

  indices_group0 = which(group_labels == 0)
  indices_group1 = which(group_labels == 1)

  X0 = data[indices_group0,]
  X1 = data[indices_group1,]

  for(j in 1:(d/2)){
    dim1 = 2*j - 1
    dim2 = 2*j

    #plot(X0[,dim1], X0[,dim2], xlim = c(0,1), ylim = c(0,1))
    plot(X0[,dim1], X0[,dim2], xlab = paste("X", dim1, sep=""), ylab = paste("X", dim2, sep=""))
    points(X1[,dim1], X1[,dim2], col = "red")
  }

  log_w_true_data = out_data$true_log_w_obs
}
