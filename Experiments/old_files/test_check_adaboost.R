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

n0 = 1000
n1 = 50

set.seed(1)
out_data = simulation_2d(n0, n1, "global_shift", 100)

data = out_data$data
group_labels = out_data$group_labels

indices_group0 = which(group_labels == 0)
indices_group1 = which(group_labels == 1)

# prep for adaboost
d = ncol(data)

labels_X = paste("X", 1:d, sep = "")

df = data.frame(y = factor(group_labels), data = data)
colnames(df)[-1] = labels_X

control = rpart.control(maxdepth = 4,cp = -1, minsplit = 0)

# run adaboost
model = ada(y ~ ., data = df, type = "real",
            control = control,iter=100, nu=0.01, bag.frac=0.5)

# evaluation on the observed data sets
prob_pred_data = predict(model, df, type = "prob")

plot(prob_pred_data[,1])
plot(log(prob_pred_data[,1] / prob_pred_data[,2] ) )
plot(log(prob_pred_data[,1] / prob_pred_data[,2] * n1 / n0) )
