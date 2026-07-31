
# (1) 2D settings

# results should be put outside of the package directory!
source("./R/models_2d.R")

user_name <- Sys.info()["user"]
location_output_root = paste(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results",sep=""))

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

# global experiment settings
n_repeat = 50

data_settings = list()

data_settings[[1]] = list("global_shift", 5000, 5000)
data_settings[[2]] = list("global_shift", 9000, 1000)
data_settings[[3]] = list("local_shift", 5000, 5000)
data_settings[[4]] = list("local_shift", 9000, 1000)
data_settings[[5]] = list("local_dispersion", 5000, 5000)
data_settings[[6]] = list("local_dispersion", 9000, 1000)

for(index_repeat in 1:n_repeat){

  for(index_settings in 1:length(data_settings)){

    data_settings_current = data_settings[[index_settings]]

    scenario = data_settings_current[[1]]
    n0 = data_settings_current[[2]]
    n1 = data_settings_current[[3]]

    set.seed(index_repeat)
    out_data = simulation_2d(n0, n1, scenario, 10)

    data = out_data$data

    log_w_true_data = out_data$true_log_w_obs

    name_output = paste(scenario, n0, n1,lambda_0, index_repeat, sep = "_")

    location_output = paste(location_output_root, "experiments_2D", scenario, sep = "/")

    saveRDS(log_w_true_data, paste(location_output, "/", name_output, "_true_log_ratio.rds", sep=""))
  }
}


########################################################################################################
# (2) 20D settings

# results should be put outside of the package directory!
user_name <- Sys.info()["user"]
location_output_root = paste(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/balancePM_results",sep=""))

source("./R/models_multi.R")

data_settings = list()
data_settings[[1]] = list("latent_location_shift", 5000, 5000, 0.2, FALSE)
data_settings[[2]] = list("latent_location_shift", 9000, 1000, 0.2, FALSE)
data_settings[[3]] = list("latent_dispersion", 5000, 5000, 0.2, FALSE)
data_settings[[4]] = list("latent_dispersion", 9000, 1000, 0.2, FALSE)

data_settings[[5]] = list("latent_location_shift", 5000, 5000, 0.2, TRUE)
data_settings[[6]] = list("latent_location_shift", 9000, 1000, 0.2, TRUE)
data_settings[[7]] = list("latent_dispersion", 5000, 5000, 0.2, TRUE)
data_settings[[8]] = list("latent_dispersion", 9000, 1000, 0.2, TRUE)

for(index_repeat in 1:n_repeat){

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
    log_w_true_data = out_data$true_log_w_obs

    location_output = paste(location_output_root, "experiments_multi", scenario, sep = "/")

    if(!transform){
      name_output = paste(scenario, n0, n1, lambda_0,index_repeat, sep = "_")
    }else{
      name_output = paste(scenario, n0, n1, "transformed", lambda_0, index_repeat, sep = "_")
    }

    saveRDS(log_w_true_data, paste(location_output, "/", name_output, "_true_log_ratio.rds", sep=""))
  }
}
