
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

source("./R/models_2d_alt.R")
source("./R/utilities_for_experiment.R")

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

# global experiment settings
K_CV = 2
num_trees_max = 1000
num_trees_Bayes = 50

learn_rate = 0.01
max_depth = 4
n_min_obs_per_node = 1

n_bins = 100
alpha_cutpoint = 1

size_burnin = 2000
size_backfitting = 500
thin = 1

lambda_0 = 1

probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)

my_quantile = function(X){
  return(quantile(X, probs = probs_output_quantile))
}

# list of settings
data_settings = list()
data_settings[[1]] = list(1, 1000, 1000)
data_settings[[2]] = list(1, 2000, 500)
data_settings[[3]] = list(2, 1000, 1000)
data_settings[[4]] = list(2, 2000, 500)
data_settings[[5]] = list(3, 1000, 1000)
data_settings[[6]] = list(3, 2000, 500)

#data_settings[[1]] = list(3, 1000, 1000)

#index_settings = 1

#data_settings[[1]] = list(1, 1000, 1000)
#data_settings[[2]] = list(2, 1000, 1000)
#data_settings[[3]] = list(3, 1000, 1000)


for(index_settings in 1:length(data_settings)){

  # simulate data
  data_settings_current = data_settings[[index_settings]]

  scenario = data_settings_current[[1]]
  n0 = data_settings_current[[2]]
  n1 = data_settings_current[[3]]

  set.seed(10)
  out_data = simulation_2d(n0, n1, scenario)

  data = out_data$data

  group_labels = out_data$group_labels
  indices_0 = which(group_labels == 0)
  indices_1 = which(group_labels == 1)

  log_w_true_grid = out_data$true_log_w_obs

  range_x = range(data[,1])
  range_y = range(data[,2])
  plot(data[indices_0,1],data[indices_0,2],xlim=range_x,ylim=range_y, xlab = "x1", ylab = "x2")
  points(data[indices_1,1],data[indices_1,2],xlim=range_x,ylim=range_y,col= "red")

  # estimation

  ########################################################################################################
  # method 1: proposed boosting (gradient based)
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
                                                quiet = FALSE
  )

  if(K_CV > 0){
    loss_grad = colMeans(result_estimation$loss_CV_store)
  }
  log_ratio_boosting_grad_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)

  ########################################################################################################
  # method 2: proposed boosting (Hellinger based)
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
                                                         quiet = FALSE
  )

  if(K_CV > 0){
    loss_hell = colMeans(result_estimation$loss_CV_store)
  }
  log_ratio_boosting_hell_data = balance_to_log_ratio(result_estimation$balance_weight_boosting_data)

  ########################################################################################################
  # method 3: Bayesian back-fitting
  result_estimation = estimate_balancing_weight_Bayes(data = data,
                                                group_labels = group_labels,
                                                num_trees = num_trees_Bayes,
                                                max_resol = 0,
                                                learn_rate = 0.01,
                                                n_bins = n_bins,
                                                alpha_cutpoint = alpha_cutpoint,
                                                n_min_obs_per_node = n_min_obs_per_node,
                                                use_gradient = F,
                                                size_burnin = size_burnin,
                                                size_backfitting = size_backfitting,
                                                thin = thin,
                                                lambda_0 = lambda_0,
                                                output_BART_ensembles = TRUE,
                                                quiet = FALSE,
                                                update_lambda = FALSE
  )

  # result of the BART
  log_ratio_BART_data = balance_to_log_ratio(result_estimation$balance_weight_BART_data)
  log_ratio_BART_data_mean = rowMeans(log_ratio_BART_data)

  log_ratio_BART_data_quantiles = apply(log_ratio_BART_data, 1, my_quantile)

  ########################################################################################################
  # visualize the result
  if(K_CV > 0){
    par(mfrow = c(1,2))
    plot(loss_grad, type = "l", xlab = "# trees", ylab = "Loss", main = "Boosting (Gradient)")
    plot(loss_hell, type = "l", xlab = "# trees", ylab = "Loss", main = "Boosting (Hellinger)")
    par(mfrow = c(1,1))
  }

  plot(density(log_ratio_BART_data_mean))
  lines(density(log_ratio_boosting_hell_data), col = "red")

  labels_method = c("True",
                    "Boosting (Gradient)",
                    "Boosting (Hellinger)",
                    "Bayes",
                    "Lower (2.5%)",
                    "Upper (97.5%)")

  n_methods = length(labels_method)

  grid.points = data

  densities.df = data.frame(x1=rep(grid.points[,1],n_methods),
                            x2=rep(grid.points[,2],n_methods),
                            log_weight = c(log_w_true_grid,
                                           log_ratio_boosting_grad_data,
                                           log_ratio_boosting_hell_data,
                                           log_ratio_BART_data_mean,
                                           log_ratio_BART_data_quantiles[1,],
                                           log_ratio_BART_data_quantiles[7,]
                            ),
                            method = factor(rep(labels_method, each = nrow(grid.points)),levels = labels_method)
  )

  #min_true = min(densities.df$log_weight)
  #max_true = max(densities.df$log_weight)

  min_all = -10
  max_all = 10

  scalle1 = 1

  plot_namme = paste("Case ", scenario, ", n_p = ", n0, ", n_q = ", n1, ", lambda0 = ", round(lambda_0), sep = "")

  output_location = "C:/Users/naway/Dropbox/Rcpp_experiments/active_programs/balancePM_results/experiments_2D_alt"
  png(filename = paste(output_location, "/Case_", scenario, "_np_", n0, "_nq_", n1,"_lambda0_",lambda_0,".png", sep = ""), width = 1000, height = 600)

  print(ggplot(densities.df, aes(x = x1, y = x2, color = log_weight)) +
    geom_point(size=3, alpha=0.75) +
    scale_color_gradientn(
      colours = c("darkblue",
                  "blue",
                  "deepskyblue",
                  "cyan",
                  "#7FFFFF",
                  "white",
                  "white",
                  "white",
                  "#FFFF99",
                  "yellow",
                  "orange",
                  "red",
                  "darkred"),
      values = scales::rescale(c(min_all,
                                 min_all * 4/5,
                                 min_all * 3/5,
                                 min_all * 2/5,
                                 min_all * 1/5,
                                 min_all/10000,
                                 0,
                                 max_all/10000,
                                 max_all * 1/5,
                                 max_all * 2/5,
                                 max_all * 3/5,
                                 max_all * 4/5,
                                 max_all)),  # 値の位置指定（0中心）
      limits = c(min_all, max_all),   # グラデーションはこの範囲に限定
      oob = scales::squish
    ) +
    facet_wrap(~method, nrow = 2)+
    labs(color = "log(density ratio)")  +
    ggtitle(paste("Case ", scenario, ", n_p = ", n0, ", n_q = ", n1, sep = ""))
  )

    dev.off()

}
