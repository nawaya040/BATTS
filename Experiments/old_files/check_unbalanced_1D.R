
# goal (2025/04/07):
# update the code so that we can evaluate the importance weight for any input point
# not limited to the observed points

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

user_name <- Sys.info()["user"]

# Rcpp and R programs
#sourceCpp(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/main.cpp",sep=""))
#sourceCpp(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/post.cpp",sep=""))
#
#source(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/adaboost_functions.R",sep=""))
#source(paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/models_2d.R",sep=""))


#settings
n0 = 5000
n1 = 200

# parameters
means = c(-0.5, 0.5)
sig2 = 1

w_unif = 0.1

range_unif = c(means[1]-4*sqrt(sig2), means[2]+4*sqrt(sig2))
dens_unif = 1 / (range_unif[2] - range_unif[1])

# obtain the true density ratio
eval_points = matrix(seq(range_unif[1], range_unif[2],by=0.01),ncol=1)
n_eval = length(eval_points)
dens_ratio_true = numeric(n_eval)

for(i in 1:n_eval){
  dens_ratio_true[i] = (w_unif * dens_unif + (1-w_unif) * dnorm(eval_points[i],means[1], sqrt(sig2))) /
                         (w_unif * dens_unif + (1-w_unif) * dnorm(eval_points[i],means[2], sqrt(sig2)))
}

log_dens_ratio_true = log(dens_ratio_true)

n_repeat = 10

curve_store_grad = matrix(NA, nrow = n_repeat, ncol = n_eval)
curve_store_hell = matrix(NA, nrow = n_repeat, ncol = n_eval)
curve_store_ada = matrix(NA, nrow = n_repeat, ncol = n_eval)

# generate the data
for(index_repeat in 1:n_repeat){

  set.seed(index_repeat)

  group_labels = c(rep(0,n0), rep(1,n1))

  data0 = matrix(NA, nrow = n0, ncol = 1)

  for(i in 1:n0){
    if(runif(1) < w_unif){
      data0[i,1] = runif(1, min = range_unif[1], max = range_unif[2])
    }else{
      data0[i,1] = rnorm(1, means[1], sqrt(sig2))
    }
  }

  data1 = matrix(NA, nrow = n1, ncol = 1)

  for(i in 1:n1){
    if(runif(1) < w_unif){
      data1[i,1] = runif(1, min = range_unif[1], max = range_unif[2])
    }else{
      data1[i,1] = rnorm(1, means[2], sqrt(sig2))
    }
  }

  data = rbind(data0, data1)

  indices_0 = which(group_labels==0)

  # visualization
  plot(density(data[which(group_labels==0),1]), xlab = "x1", ylab = "x2", main = "observations")
  lines(density(data[which(group_labels==1),1]),col = "red")
  points(data0,rep(0,n0))
  points(data1,rep(0,n1),col="red")

  # gradiend-based

  result_grad = estimate_balancing_weight(data,
                           group_labels,
                           num_trees = 500,
                           K_CV = 5 ,
                           use_gradient = TRUE,
                           quiet = TRUE
  )

  result_eval_grad = eval_balance_weight(result_grad, matrix(eval_points, ncol=1), BART_result = FALSE)

  curve_store_grad[index_repeat,] = log(result_eval_grad$balancing_weight_boosting) * 2

  # hellinger-based

  result_hell = estimate_balancing_weight(data,
                                          group_labels,
                                          num_trees = 500,
                                          K_CV = 5 ,
                                          use_gradient = TRUE,
                                          quiet = TRUE
  )

  result_eval_hell = eval_balance_weight(result_hell, matrix(eval_points, ncol=1), BART_result = FALSE)

  curve_store_hell[index_repeat,] = log(result_eval_hell$balancing_weight_boosting) * 2

  #num_trees_ada = 500
  #df = data.frame(y = factor(group_labels), data)
  #
  #control = rpart.control(maxdepth = 2,cp = -1, minsplit = 0)
  #model = ada(y ~ ., data = df, type = "real", control = control,iter=num_trees_ada, nu=0.01)
  #
  #newdata = data.frame(data = eval_points)
  #prob_pred = predict(model, newdata, type = "prob")
  #
  #log_w_hat_ada = log(prob_pred[,1] / prob_pred[,2] * n1 / n0)
  #
  #curve_store_ada[index_repeat,] = log_w_hat_ada
}



#plot(eval_points, prob_pred[,1], type = "l")

# evaluate the balancing weights
#result_eval = eval_balance_weight(result_boosting, eval_points)

#range_plot = range(c(log_dens_ratio_true, log(balance_estimated_grad) * 2, log(balance_estimated_hell) * 2, log_w_hat_ada))
range_plot = range(c(log_dens_ratio_true, colMeans(curve_store_grad), colMeans(curve_store_hell)))

lwd_plot = 2

plot(eval_points, log_dens_ratio_true, type = "l", ylim = range_plot, xlab = "x", ylab = "log(dens ratio)",
     main = paste("n0=",n0,", n1=",n1,sep=""), lwd = lwd_plot, lty = 3)
lines(eval_points, colMeans(curve_store_grad), col="darkgreen", lwd = lwd_plot, lty = 2)
lines(eval_points, colMeans(curve_store_hell), col="red", lwd = lwd_plot)
#lines(eval_points, colMeans(curve_store_ada), col="blue")

#result = list(n0, n1, curve_store_grad, curve_store_hell)
#saveRDS(result, paste("C:/Users/",user_name,"/Dropbox/Rcpp_experiments/active_programs/Adaboost_20241206/result_balanced.rds",sep=""))


