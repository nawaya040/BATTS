library(devtools)
install_github("nawaya040/boostPM")

data_train = read.csv("C:/Users/naway/Dropbox/Rcpp_experiments/data/Duke_Microbiome/sample_train.csv", header=F)
plot(data_train$V18)

d = ncol(data_train)

indices_no_var = which(sapply(1:d, function(i) var(data_train[,i])) == 0)
indices_all = 1:d
indices_with_var = indices_all[-indices_no_var]
data_train_to_use = data_train[indices_with_var]

library(boostPM)

#Run boosting
out = boosting(# parameters for boosting
  data = data_train_to_use, #data = n X d matrix
  add_noise = T, # add uniform noises if there are tied values
  Omega = NULL,
  ntree_max_marginal = 500, # # trees per dimension used in the first stage
  ntree_max_dependence = 5000, # # trees used in the second stage
  c0 = 0.1, # c0 = global scale of the learning parameter
  gamma = 0.5, # gamma = stronger regularization for small nodes
  max_resol = 10, # maximum resolution (depth) of trees
  min_obs = 5, # if # obs in a node > min_obs, this node is no longer split
  early_stop = c(1e-5,50), # if it is (1e-5, 50), this means we move to the next step
  # when the average improvement given by the recent 50 trees is less than 1e-5
  nbins = 100, # # bins (n_bins-1 = # grid points)
  max_n_var = d, # this is an experimental one so should be set to d
  # parameters for the PT-based weak learner
  alpha = 0.9, # prior prob of dividing a node = alpha * (1 + depth)^beta
  beta = 0.0,
  precision = 1.0 # precision of the theta prior
)

#Simulate from the estimated distribution
simulated.data = simulation_b(list_boosting = out, # simply use the output of the boosting function
                              size = 1000 # size of simulation
)

d1 = 9
d2 = 71

simulated.data_adjusted = simulated.data
simulated.data_adjusted[which(simulated.data<0)]=0

plot(data_train_to_use[,d1], data_train_to_use[,d2], xlab = "x1", ylab = "x2",main = "observations")
points(simulated.data_adjusted[,d1], simulated.data_adjusted[,d2],xlab="x1",ylab="x2", main="simulated data", col="red")

simulated.data_out = matrix(0, nrow = 1000, ncol = d)
simulated.data_out[,indices_with_var] = simulated.data_adjusted

write.table(simulated.data_out, file = "sample_boostPM.csv", sep = ",", row.names = FALSE, col.names = FALSE)
