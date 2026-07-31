
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

library(pracma)

source("./R/models_2D.R")
source("./R/utilities_for_experiment.R")

data_settings = list()
data_settings[[1]] = list("global_shift", 5000, 5000)
data_settings[[2]] = list("local_shift", 5000, 5000)
data_settings[[3]] = list("local_dispersion", 5000, 5000)

scenario_names = c("1. Global shift", "2. Local shift", "3. Local dispersion")

index_repeat = 1
n_grid_per_dim = 100

data_store = c()
group_labels_store = c()
scenario_store = c()

for(index_settings in 1:3){

  data_settings_current = data_settings[[index_settings]]

  scenario = data_settings_current[[1]]
  n0 = data_settings_current[[2]]
  n1 = data_settings_current[[3]]

  set.seed(index_repeat)
  out_data = simulation_2d(n0, n1, scenario, n_grid_per_dim)

  data = out_data$data
  group_labels = out_data$group_labels

  data_store = rbind(data_store, data)
  group_labels_store = c(group_labels_store, group_labels)

  scenario_store = c(scenario_store, rep(scenario_names[index_settings], nrow(data)))

  #random_order = sample(1:nrow(data))

  #data = data[random_order,]
  #group_labels = group_labels[random_order]

  #indices_0 = which(group_labels == 0)
  #indices_1 = which(group_labels == 1)

  #plot(data[indices_0,1], data[indices_0,2], xlab = "x1", ylab = "x2", main = scenario,
  #     xlim = range(data[,1]), ylim = range(data[,2]))
  #points(data[indices_1,1], data[indices_1,2], col = "red")
}

data.df = data.frame(x1 = data_store[,1], x2 = data_store[,2], group_labels = group_labels, scenario = scenario_store)

data.df = data.df[sample(1:nrow(data.df)),]


ggplot(data.df, aes(x = x1, y = x2, color = factor(group_labels))) +
  geom_point(size = 2, alpha=0.5) +
  scale_color_manual(values = c("0" = "black", "1" = "red")) +
  labs(color = "Label") +
  facet_wrap(~scenario, scales = "free") +
  guides(color = "none") +
  theme_minimal()
