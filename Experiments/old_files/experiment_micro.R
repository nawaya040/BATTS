
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
library(ada)
library(densratio)

library(foreach)
library(doParallel)

library(balancePM)

library(phyloseq)

library(GGally)

source("./R/models_2d.R")
source("./R/utilities_for_experiment.R")

balance_to_log_ratio = function(x){
  return(log(x)*2)
}

# import the data
user_name <- Sys.info()["user"]
loc_data = paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/data/DIABIMMUNE", sep = "")
DIABIMMUNE_processed = readRDS(paste(loc_data, "DIABIMMUNE_processed.rds", sep = "/"))

ps_combined = DIABIMMUNE_processed$ps_combined
data = DIABIMMUNE_processed$ra_mat

#flag = "post_seroconversion"
flag = "case_control"

if(flag == "post_seroconversion"){
  group_labels = DIABIMMUNE_processed$post_seroconversion
}

if(flag == "case_control"){
  group_labels = DIABIMMUNE_processed$case_control
}


sum(group_labels == 0)
sum(group_labels == 1)



group_labels_factor = factor(group_labels)

#d1 = 2
#d2 = 3
#
#plot(data[which(group_labels==0),d1], data[which(group_labels==0),d2])
#points(data[which(group_labels==1),d1], data[which(group_labels==1),d2], col = "red")

# run MCMC

# calculate quarantines
probs_output_quantile = c(0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975)
my_quantile = function(X){
  return(quantile(X, probs = probs_output_quantile))
}


num_trees_Bayes = 200

learn_rate_for_Bayes = 0.01
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

result_estimation = estimate_balancing_weight_Bayes(data = data,
                                                    group_labels = group_labels,
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
log_ratio_BART_data_means = rowMeans(log_ratio_BART_data)

# obtain the quarantines
log_ratio_BART_data_quantiles = apply(log_ratio_BART_data, 1, my_quantile)


# PCoA
library("ggplot2")
library("RColorBrewer")

ord = ordinate(ps_combined, 'PCoA', 'bray')
plot_ordination(ps_combined, ord)

dim = 3

out_df = data.frame(ord$vectors[,1:dim], group_labels = group_labels_factor, log_ratio = log_ratio_BART_data_means,
                    lower = log_ratio_BART_data_quantiles[1,],
                    upper = log_ratio_BART_data_quantiles[7,])

out_df = out_df[sample(1:nrow(out_df)),]

ggpairs(out_df,
        columns = 1:dim,
        mapping = aes(color = group_labels,
                     shape = group_labels))+
  ggtitle(flag) +  # タイトルを追加
  theme(plot.title = element_text(hjust = 0.5))  +
  theme_minimal() # Columns

ggpairs(out_df,
        columns = 1:dim,  # x, y, z
        mapping = aes(color = log_ratio, shape = group_labels),
        lower = list(continuous = wrap("points", alpha = 0.9, size = 2)),
        upper = list(continuous = wrap("points", alpha = 0.9, size = 2)),
        diag = list(continuous = wrap("blankDiag"))) +
  scale_color_gradient2(low = "blue",high = "red", mid = "white", midpoint=0) +
  theme(legend.position = "right") +
  ggtitle("posterior mean") +  # タイトルを追加
  theme(plot.title = element_text(hjust = 0.5))  +
  theme_minimal() # Columns



ggpairs(out_df,
        columns = 1:dim,  # x, y, z
        mapping = aes(color = lower, shape = group_labels),
        lower = list(continuous = wrap("points", alpha = 0.9, size = 2)),
        upper = list(continuous = wrap("points", alpha = 0.9, size = 2)),
        diag = list(continuous = wrap("blankDiag"))) +
  scale_color_gradient2(low = "blue",high = "red", mid = "white", midpoint=0) +
  theme(legend.position = "right") +
  ggtitle("Lower quantiles") +  # タイトルを追加
  theme(plot.title = element_text(hjust = 0.5))  +
  theme_minimal() # Columns



ggpairs(out_df,
        columns = 1:dim,  # x, y, z
        mapping = aes(color = upper, shape = group_labels),
        lower = list(continuous = wrap("points", alpha = 0.9, size = 2)),
        upper = list(continuous = wrap("points", alpha = 0.9, size = 2)),
        diag = list(continuous = wrap("blankDiag"))) +
  scale_color_gradient2(low = "blue",high = "red", mid = "white", midpoint=0) +
  theme(legend.position = "right") +
  ggtitle("Upper quantiles") +  # タイトルを追加
  theme(plot.title = element_text(hjust = 0.5))  +
  theme_minimal() # Columns


