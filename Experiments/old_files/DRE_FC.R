# libraries we (may) use in this experiment
library(Rcpp)
library(RcppArmadillo)
library(parallel)


library(devtools)
library(usethis)

library(balancePM)

library(foreach)
library(doParallel)

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

# import the data
user_name = Sys.info()["user"]

setwd(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/data/masscytometry_data",sep=""))
data_bc = as.matrix(read.csv("Exp 1_Patient #1_bc.csv"))
data_pt = as.matrix(read.csv("Exp 1_Patient #1_pt.csv"))

n_bc = nrow(data_bc)
n_pt = nrow(data_pt)

data_bc_sub = data_bc[sample(1:n_bc, size = round(5000)),]
data_pt_sub = data_pt[sample(1:n_pt, size = round(5000)),]

data = rbind(data_bc_sub, data_pt_sub)
labels = c(rep(1,5000), rep(2,5000))

ramdom_order = sample(1:10000)

library(Rtsne)
tsne_res <- Rtsne(data, perplexity = 30)
plot(tsne_res$Y[ramdom_order,], col = labels[ramdom_order], pch = 1)

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

result_estimation = estimate_balancing_weight_Bayes(data = data,
                                                    group_labels = labels-1,
                                                    num_trees = num_trees_Bayes,
                                                    max_resol = max_depth_for_Bayes,
                                                    learn_rate = learn_rate_for_Bayes,
                                                    n_bins = n_bins,
                                                    alpha_cutpoint = alpha_cutpoint,
                                                    n_min_obs_per_node = n_min_obs_per_node,
                                                    use_gradient = F,
                                                    size_burnin = size_burnin,
                                                    size_backfitting = size_backfitting,
                                                    thin = thin,
                                                    lambda_0 = lambda_0,
                                                    output_BART_ensembles = TRUE,
                                                    quiet = T,
                                                    update_lambda = FALSE
)

# result of the BART
log_ratio_BART_data = balance_to_log_ratio(result_estimation$balance_weight_BART_data)

probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)
my_quantile = function(X){
  return(quantile(X, probs = probs_output_quantile))
}

# obtain the quarantines
log_ratio_BART_data_quantiles = apply(log_ratio_BART_data, 1, my_quantile)

library(ggplot2)
rtsne_df <- data.frame(x1 = tsne_res$Y[ramdom_order,1], x2 = tsne_res$Y[ramdom_order,2],
                       log_ratio = rowMeans(log_ratio_BART_data)[ramdom_order],
                       lower = log_ratio_BART_data_quantiles[1,ramdom_order],
                       upper = log_ratio_BART_data_quantiles[7,ramdom_order])
plot(rtsne_df$log_ratio)

ggplot(rtsne_df, aes(x = x1, y = x2, color = log_ratio)) +
  geom_point() +
  scale_color_gradient2(low = "blue",high = "red", mid = "white", midpoint=0)+
  theme_minimal()

ggplot(rtsne_df, aes(x = x1, y = x2, color = lower)) +
  geom_point() +
  scale_color_gradient2(low = "blue",high = "red", mid = "white", midpoint=0)+
  theme_minimal()

ggplot(rtsne_df, aes(x = x1, y = x2, color = upper)) +
  geom_point() +
  scale_color_gradient2(low = "blue",high = "red", mid = "white", midpoint=0)+
  theme_minimal()

plot(density(rowMeans(log_ratio_BART_data)[1:5000]), xlim = c(-10,6))
lines(density(rowMeans(log_ratio_BART_data)[5001:10000]), col = "red")
