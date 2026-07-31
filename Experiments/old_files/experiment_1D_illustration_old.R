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

windowsFonts(Helvetica = windowsFont("Helvetica"),
             Arial = windowsFont("Arial"))

# settings for simulation (a simple Gaussian model should be enough)
mu0 = 0
mu1 = 1
sig20 = 1^2
sig21 = 1.5^2

BD_true = 1/4 * (mu0 - mu1)^2 / (sig20 + sig21) +
  1/2 * log((sig20 + sig21) / (2 * sqrt(sig20 * sig21)))
BC_true = exp(-BD_true)


# the other settings
num_trees_Bayes = 200

learn_rate_for_Bayes = 0.01
max_depth_for_Bayes = 0
n_min_obs_per_node = 1

n_bins = 1000
alpha_cutpoint = 1

size_burnin = 1000
size_backfitting = 2000
thin = 1

lambda_0 = 5

n_grid_per_dim = 100

quant_probs = seq(0.005, 0.995, by = 0.005)
n_probs = length(quant_probs)
c_probs = 1 - quant_probs[1:(floor(n_probs/2))] * 2

setting_list = list()
# setting_list[[1]] = c(500, 500)
# setting_list[[2]] = c(100, 900)

scale = 5
setting_list[[1]] = c(500, 500) * scale
setting_list[[2]] = c(100, 900) * scale

for(index_setting in 1:length(setting_list)){

  n0 = setting_list[[index_setting]][1]
  n1 = setting_list[[index_setting]][2]

  set.seed(1)

  # simulate the data
  data = matrix(c(rnorm(n0, mu0, sqrt(sig20)), rnorm(n1, mu1, sqrt(sig21))),ncol = 1)

  grid_points = seq(-2.5, 3.5, length.out = 100)

  # check the truth
  log_dens_ratio_true = dnorm(grid_points, mu0, sqrt(sig20), log = TRUE) -
    dnorm(grid_points, mu1, sqrt(sig21), log = TRUE)

  # dens_ratio_true = dnorm(grid_points, mu0, sqrt(sig20), log = FALSE) /
  #   dnorm(grid_points, mu1, sqrt(sig21), log = FALSE)

  # plot(grid_points, log_dens_ratio_true, type = "l")
  # plot(grid_points, dens_ratio_true, type = "l")

  # estimation with BART
  result_estimation = batts(data = data,
                            group_labels = c(rep(0,n0), rep(1,n1)),
                            num_trees = num_trees_Bayes,
                            # n_bins = n_bins,
                            size_burnin = size_burnin,
                            size_backfitting = size_backfitting,
                            output_BART_ensembles = TRUE,
                            lambda_0 = lambda_0,
                            quiet = F,
                            update_lambda = FALSE
  )

  tau_inv_result = result_estimation$omega_store^(-1)

  eval_BAT = eval_balance_weight(list_result = result_estimation,
                                 eval_points = matrix(grid_points),
                                 is_Bayes = T)

  log_ratio_grid_post_mean = rowMeans(log(eval_BAT$balancing_weight_BART) * 2)
  log_ratio_grid_true = dnorm(grid_points, mu0, sqrt(sig20), log = TRUE) -
    dnorm(grid_points, mu1, sqrt(sig21), log = TRUE)

  CIs =  apply(log(eval_BAT$balancing_weight_BART) * 2,
               1, quantile, probs = quant_probs)

  # visualize the result from several aspects
  range_plot = range(grid_points)

  # (1) estimated log ratios v.s. true
  plot_df = data.frame(x = grid_points,
                       true_log_dens = log_ratio_grid_true,
                       mean = log_ratio_grid_post_mean,
                       lower = CIs["2.5%",],
                       upper = CIs["97.5%",]
                       )

  plot_df_log <- plot_df %>%
    pivot_longer(cols = -x, names_to = "series", values_to = "y")

  p1 = ggplot(plot_df_log, aes(x = x, y = y, color = series, linetype = series)) +
    geom_line(linewidth = 1)+
    scale_color_manual(values = c(
      true_log_dens = "grey50",   # 真値: グレー
      mean = "black",    # 事後平均: 黒
      lower  = "blue",
      upper  = "blue" # 分位点: 同じ色
    )) +
    scale_linetype_manual(values = c(
      true_log_dens = "dotted",   # 真値: 点線
      mean = "solid",    # 事後平均: 実線
      lower  = "dashed",   # 分位点: 破線
      upper  = "dashed"
    )) +
    theme_bw() + theme(legend.position = "none") + labs(x = "x", y = "log (density ratio)") +
    coord_cartesian(ylim = c(-4.5, 1.5))


  # 3. estimated BC
  p2 =ggplot(data.frame(tau_inv = tau_inv_result), aes(x = tau_inv)) +
    geom_density(linewidth = 1) +
    geom_vline(xintercept = BC_true,
               color = "grey50", linetype = "dashed", linewidth = 1) +
    theme_bw() +
    labs(x = expression(tau^{-1}), y = "Posterior density") +
    theme(legend.position = "none")
  # + coord_cartesian(xlim = c(0.75, 1.15), ylim = c(0.0, 13.5)) # add if sample size is smaller

  print(p1 + p2)

}
