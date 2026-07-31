
# goal (2025/04/08):
# check how the credible bands are created based on the BART
# using simple 2D examples

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

user_name <- Sys.info()["user"]

# Rcpp and R programs

#devtools::document()
#devtools::load_all()

#library(balancePM)

#sourceCpp(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/main.cpp",sep=""))
#sourceCpp(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/post.cpp",sep=""))
#
#source(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/adaboost_functions.R",sep=""))
#source(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/models_2d.R",sep=""))

# the code for simulating data is imported directly
# because we may edit the settings very often
source("./R/models_2d.R")
source("./R/utilities_for_experiment.R")

# function to compute the squared error
compute_sq_error = function(log_w_estimated, log_w_true_true){
  out = mean((log_w_estimated - log_w_true_true)^2)
  return(out)
}

#settings
n0 = 2000
n1 = 2000

d = 2

K_CV = 5

n = n0+n1

scenario = "global_shift"

n_grid_per_dim = 100

# generate the data and generate the grid points
set.seed(10)
out_data = simulation_2d(n0, n1, scenario, n_grid_per_dim)

grid.points = out_data$grid_points
n_eval = nrow(grid.points)

data = out_data$data
group_labels = out_data$group_labels
log_w_true_obs = out_data$true_log_w_obs
log_w_true_grid = out_data$true_log_w_grid

# visualization
plot(data[which(group_labels==0),1], data[which(group_labels==0),2], xlab = "x1", ylab = "x2", main = scenario)
points(data[which(group_labels==1),1], data[which(group_labels==1),2], col = "red")

num_trees = 1000

result_boosting_hell = estimate_balancing_weight(data = data,
                                                 group_labels = group_labels,
                                                 num_trees = num_trees,
                                                 K_CV = K_CV,
                                                 use_gradient = T,
                                                 size_burnin = 100,
                                                 size_backfitting = 200,
                                                 thin = 2,
                                                 output_BART_ensembles = TRUE,
                                                 quiet = FALSE
)

plot(colMeans(result_boosting_hell$loss_CV_store))

balance_estimated_result = eval_balance_weight(result_boosting_hell, grid.points, T)
balance_estimated_hell = balance_estimated_result$balancing_weight_boosting
balance_estimated_bart = balance_estimated_result$balancing_weight_BART

# MSE (average taken over the observed points)
compute_sq_error(log(result_boosting_hell$balance_weight_boosting_data)*2 , log_w_true_obs)
compute_sq_error(rowMeans(log(result_boosting_hell$balance_weight_BART_data))*2 , log_w_true_obs)

# MSE (average taken over the uniformly distributed grid points)
# approximation to L2 distance in the functional space
compute_sq_error(log(balance_estimated_hell)*2 , log_w_true_grid)
compute_sq_error(rowMeans(log(balance_estimated_bart))*2 , log_w_true_grid)

#################################################################################################
# how about adaboost?
library(ada)

result_ada = ada_log_weight(data,
              group_labels,
              grid.points,
              K_CV = 5,
              num_trees_max = 500,
              num_trees_min = 100,
              maxdepth = 4,
              learn_rate = 0.01
)

log_w_hat_ada_data = result_ada$log_w_hat_ada_data
log_w_hat_ada_grid = result_ada$log_w_hat_ada_grid

compute_sq_error(log_w_hat_ada_data , log_w_true_obs)
compute_sq_error(log_w_hat_ada_grid, log_w_true_grid)

#################################################################################################
# try densratio
library(densratio)
X0 = data[which(group_labels==0),]
X1 = data[which(group_labels==1),]

result_KLIEP = densratio(X0, X1, method = "KLIEP", verbose = FALSE)
log_w_hat_KLIEP_data = log(result_KLIEP$compute_density_ratio(data))
log_w_hat_KLIEP_grid = log(result_KLIEP$compute_density_ratio(grid.points))

result_uLSIF = densratio(X0, X1, method = "uLSIF", verbose = FALSE)
log_w_hat_uLSIF_data = log(result_uLSIF$compute_density_ratio(data))
log_w_hat_uLSIF_grid = log(result_uLSIF$compute_density_ratio(grid.points))

compute_sq_error(log_w_hat_uLSIF_data , log_w_true_obs)
compute_sq_error(log_w_hat_uLSIF_grid, log_w_true_grid)
#################################################################################################
# visualization

lower = sapply(1:n_eval, function(i) quantile(balance_estimated_bart[i,], probs = c(0.025)))
upper = sapply(1:n_eval, function(i) quantile(balance_estimated_bart[i,], probs = c(0.975)))

#Visualize the estimated log-importance weights
plot_labels = c("true log weight", "boosting", "ada", "uLSIF","post mean", "post lower", "post upper")
n_methods = length(plot_labels)

densities.df = data.frame(x1=rep(grid.points[,1],n_methods),
                          x2=rep(grid.points[,2],n_methods),
                          log_weight = c(log_w_true_grid,
                                         log(balance_estimated_hell)*2,
                                         log_w_hat_ada_grid,
                                         log_w_hat_uLSIF_grid,
                                         rowMeans(log(balance_estimated_bart) * 2),
                                         log(lower)*2,
                                         log(upper)*2),
                          method = factor(rep(plot_labels, each = nrow(grid.points)),levels = plot_labels)
                          )

min_true = min(log_w_true_grid)
max_true = max(log_w_true_grid)

min_all = min(densities.df$log_weight)
max_all = max(densities.df$log_weight)

scalle1 = 1

ggplot(densities.df, aes(x = x1, y = x2, fill = log_weight)) +
  geom_tile() +
  scale_fill_gradientn(
    colours = c("darkblue", "blue", "whitesmoke","whitesmoke","whitesmoke", "red", "darkred"),
    values = scales::rescale(c(-Inf, min_true/1.0, min_true/5, 0, max_true/5, max_true/1., Inf)),  # 値の位置指定（0中心）
    limits = c(min_true, max_true),   # グラデーションはこの範囲に限定
    oob = scales::squish
  ) +
  facet_wrap(~method)


