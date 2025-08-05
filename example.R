
# packages
library(mvtnorm)
library(ggplot2)

# the functions output the square root of density ratios by default
# make a function to transform them to the log-density ratios
balance_to_log_ratio = function(x){
  return(log(x)*2)
}

calc_quantiles = function(X){
  return(quantile(X, probs = c(0.025,0.975)))
}

# generate the data set
set.seed(1)

n0 = 1000
n1 = 1000

X0 = rmvnorm(n0, mean=c(-0.5, -0.5))
X1 = rmvnorm(n1, mean=c(0.5, 0.5))

data = rbind(X0, X1)
group_labels = c(rep(0,n0), rep(1,n1))

# (1) estimate the density ratio with the gradient boosting
result_boosting = estimate_balancing_weight_boosting(data = data,
                                                     group_labels = group_labels,
                                                     num_trees = 500,
                                                     K_CV = 2,
                                                     max_resol = 4,
                                                     learn_rate = 0.01,
                                                     n_bins = 32,
                                                     margin_scale = 0.1,
                                                     use_gradient = T,
                                                     quiet = F
)

log_ratio_boosting = balance_to_log_ratio(result_boosting$balance_weight_boosting_data)

# (2) estimate the density ratio with the Bayesian additive trees
result_BAT = estimate_balancing_weight_Bayes(data = data,
                                                    group_labels = group_labels,
                                                    num_trees = 200,
                                                    n_bins = 32,
                                                    margin_scale = 0.1,
                                                    size_burnin = 500,
                                                    size_backfitting = 500,
                                                    output_BART_ensembles = T,
                                                    quiet = F
)

log_ratio_BAT = balance_to_log_ratio(result_BAT$balance_weight_BART_data)
log_ratio_BAT_mean = rowMeans(log_ratio_BAT)
log_ratio_BAT_quantiles = apply(log_ratio_BAT,1,calc_quantiles)

# visualize the estimated log-densities
methods = c("GB", "BAT","lower","upper")

ratio_df = data.frame(x = rep(data[,1],4),
                     y = rep(data[,2],4),
                     log_ratio = c(log_ratio_boosting,
                                   log_ratio_BAT_mean,
                                   log_ratio_BAT_quantiles[1,],
                                   log_ratio_BAT_quantiles[2,]),
                     method = factor(rep(methods, each = n0+n1), levels = methods)
                     )

max_abs = max(abs(ratio_df$log_ratio))

ggplot(ratio_df, aes(x = x, y = y, color = log_ratio)) +
  geom_point(size=2, alpha=0.75) +
             scale_color_gradientn(
             colours = c("blue","white","red"),
             values = scales::rescale(c(-max_abs,0,max_abs)),
             limits = c(-max_abs,max_abs),
             oob = scales::squish
             ) +
  facet_wrap(~method, nrow = 2) +
  labs(color = "log(density ratio)")

# we can also evaluate the density ratios for test data
eval_points = rmvnorm(1000, mean=c(0,0), sigma = diag(c(1,1)))

eval_boosting = eval_balance_weight(result_boosting, eval_points, BART_result = F)
log_ratio_eval_boosting = balance_to_log_ratio(eval_boosting$balancing_weight_boosting)

eval_BAT = eval_balance_weight(result_BAT, eval_points, BART_result = T)
log_ratio_eval_BAT = balance_to_log_ratio(eval_BAT$balancing_weight_BART)
log_ratio_eval_BAT_mean = rowMeans(log_ratio_eval_BAT)
log_ratio_eval_BAT_quantiles = apply(log_ratio_eval_BAT,1,calc_quantiles)

# visualize the evaluated log-densities
methods = c("GB", "BAT","lower","upper")

ratio_eval_df = data.frame(x = rep(eval_points[,1],4),
                           y = rep(eval_points[,2],4),
                           log_ratio = c(log_ratio_eval_boosting,
                                        log_ratio_eval_BAT_mean,
                                        log_ratio_eval_BAT_quantiles[1,],
                                        log_ratio_eval_BAT_quantiles[2,]),
                           method = factor(rep(methods, each = n0+n1), levels = methods)
)

max_abs = max(abs(ratio_eval_df$log_ratio))

ggplot(ratio_eval_df, aes(x = x, y = y, color = log_ratio)) +
  geom_point(size=2, alpha=0.75) +
  scale_color_gradientn(
    colours = c("blue","white","red"),
    values = scales::rescale(c(-max_abs,0,max_abs)),
    limits = c(-max_abs,max_abs),
    oob = scales::squish
  ) +
  facet_wrap(~method, nrow = 2) +
  labs(color = "log(density ratio)")
