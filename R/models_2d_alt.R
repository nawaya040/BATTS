simulation_2d = function(n0, n1, scenario){

  library(mvtnorm)

  d = 2

  group_labels = c(rep(0, n0), rep(1, n1))
  n = n0 + n1
  data = matrix(NA, nrow = n, ncol = d)

  differences_log_dens = numeric(n)

  # set means and covariance
  mean0 = numeric(2)
  mean1 = numeric(2)

  Cov0 = matrix(NA, nrow = 2, ncol = 2)
  Cov1 = matrix(NA, nrow = 2, ncol = 2)

  if(scenario == 1){
    mean0[1] = 0
    mean0[2] = 0

    mean1[1] = -2
    mean1[2] = -2

    Cov0[1,] = c(1,0.5)
    Cov0[2,] = c(0.5,1)

    Cov1[1,] = c(1,0.5)
    Cov1[2,] = c(0.5,1)
  }

  if(scenario == 2){
    mean0[1] = -1
    mean0[2] = 5

    mean1[1] = -1
    mean1[2] = 5

    Cov0[1,] = c(1,0)
    Cov0[2,] = c(0,1)

    Cov1[1,] = c(1,0)
    Cov1[2,] = c(0,1)
  }

  if(scenario == 3){
    mean0[1] = -1
    mean0[2] = 5

    mean1[1] = -1
    mean1[2] = 5

    Cov0[1,] = 2 * c(1,0)
    Cov0[2,] = 2 * c(0,1)

    Cov1[1,] = 2 * c(1,-0.9)
    Cov1[2,] = 2 * c(-0.9,1)
  }

  data[which(group_labels==0),] = rmvnorm(n0, mean0, Cov0)
  data[which(group_labels==1),] = rmvnorm(n1, mean1, Cov1)

  differences_log_dens = dmvnorm(data, mean0, Cov0, log = TRUE) -
                           dmvnorm(data, mean1, Cov1, log = TRUE)

  out = list("data" = data,
             "group_labels" = group_labels,
             "true_log_w_obs" = differences_log_dens)

  return(out)
}
