simulation_multi = function(n0, n1, d, dim_dif_max, scenario){

  library(mvtnorm)

  group_labels = c(rep(0, n0), rep(1, n1))
  n = n0 + n1
  data = matrix(NA, nrow = n, ncol = d)

  differences_log_dens = numeric(n)

  if(scenario == "global_shift"){

    indices_0 = which(group_labels == 0)
    indices_1 = which(group_labels == 1)

    size_0 = length(indices_0)
    size_1 = length(indices_1)

    differences_log_dens = numeric(n)

    mean0 = -0.5
    sd0 = 1

    mean1 = 0.5
    sd1 = 1

    k = 4

    mu_vec0 = mu_vec1 = rep(mean0, d)
    mu_vec1[1:dim_dif_max] = mean1

    Sigma0 = Sigma1 = matrix(0, nrow = d, ncol = d)
    diag(Sigma0) = sd0^2
    diag(Sigma1) = sd0^2

    for(j in 1:d){
      if(j <= dim_dif_max){
        data[indices_0,j] = rnorm(size_0, mean0, sd0)
        data[indices_1,j] = rnorm(size_1, mean1, sd1)
      }else{
        data[,j] = rnorm(n, mean0, sd0)
      }
    }

    # mix uniform components
    unif_w = 0.2
    range_x = c(mean0-k*sd0, mean1+k*sd1)

    indices_unif = rbinom(n, size = 1, prob = unif_w)
    n_unif = sum(indices_unif)

    data[which(indices_unif==1), ] = matrix(runif(n_unif*d,range_x[1],range_x[2]),
                                            nrow=n_unif, ncol=d)

    # compute the log-densities and the differences
    mu_vec_0 = rep(mean0, d)
    mu_vec_1 = c(rep(mean1, dim_dif_max), rep(mean0, d-dim_dif_max))

    Sigma_0 = diag(rep(sd0^2, d))
    Sigma_1 = diag(c(rep(sd1^2, dim_dif_max), rep(sd0^2, d-dim_dif_max)))

    unif_dens = (range_x[2]-range_x[1])^(-d)
    log_densities_0 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(data, mean = mu_vec_0, sigma = Sigma_0, log = F))
    log_densities_1 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(data, mean = mu_vec_1, sigma = Sigma_1, log = F))

    differences_log_dens = log_densities_0 - log_densities_1
  }


  out = list("data" = data,
             "group_labels" = group_labels,
             "true_log_w_obs" = differences_log_dens)

  return(out)
}

simulation_multi_latent = function(n0, n1, d, scenario, unif_w = 0.2, transform = FALSE,
                                   a_beta = 0.5, b_beta = 10){

  n = n0 + n1

  group_labels = c(rep(0, n0), rep(1, n1))
  n = n0 + n1
  data = matrix(NA, nrow = n, ncol = d)

  indices_0 = which(group_labels == 0)
  indices_1 = which(group_labels == 1)

  size_0 = length(indices_0)
  size_1 = length(indices_1)

  differences_log_dens = numeric(n)

  if(scenario == "latent_location_shift_oldest"){

    Q_full = randortho(d)

    m = 4
    U = Q_full[, 1:m]

    dim(U)

    mean_vec0 = c(0,0,0,0)
    mean_vec1 = c(1,0,0,0)

    sigma0 = diag(c(1,1,1,1)^2)
    sigma1 = diag(c(1,1,1,1)^2)

    data_eps = matrix(NA, nrow = n, ncol = m)

    sd_small = 0.1

    sd_unif = 3

    range_min = -5
    range_max = 5

    for(i in 1:n){
      if(group_labels[i] == 0){
        eps = t(rmvnorm(1, mean_vec0, sigma0))
        data_eps[i,] = eps
      }else{
        eps = t(rmvnorm(1, mean_vec1, sigma1))
        data_eps[i,] = eps
      }
    }

    # indices for the elements to be replaced by uniform components
    data = data_eps %*% t(U) + matrix(rnorm(d*n,mean=0,sd=sd_small),nrow=n,ncol=d)

    # compute the log-densities and the differences
    mean0_vec_true = U %*% mean_vec0
    mean1_vec_true = U %*% mean_vec1

    sigma0_true = U %*% sigma0 %*% t(U) + sd_small^2 * diag(d)
    sigma1_true = U %*% sigma1 %*% t(U) + sd_small^2 * diag(d)

    mean_noise_vec_true = rep(0, d)
    sigma_noise_true = sd_unif^2 * U %*% t(U) + sd_small^2 * diag(d)

    log_dens_0 = dmvnorm(data, mean0_vec_true, sigma0_true, log=TRUE)
    log_dens_1 = dmvnorm(data, mean1_vec_true, sigma1_true, log=TRUE)

    differences_log_dens = log_dens_0 - log_dens_1
  }

  if(scenario == "latent_location_shift_prev"){

    Q_full = randortho(d)

    m = 4
    U = Q_full[, 1:m]

    dim(U)

    mean_vec0 = c(0,0,0,0)
    mean_vec1 = c(1,0,0,0)

    sigma0 = diag(c(1,1,1,1)^2)
    sigma1 = diag(c(1,1,1,1)^2)

    data_eps = matrix(NA, nrow = n, ncol = m)

    sd_small = 0.1

    range_min = -5
    range_max = 5

    for(i in 1:n){
      if(group_labels[i] == 0){
        eps = t(rmvnorm(1, mean_vec0, sigma0))
        data_eps[i,] = eps
      }else{
        eps = t(rmvnorm(1, mean_vec1, sigma1))
        data_eps[i,] = eps
      }
    }

    data = data_eps %*% t(U) + matrix(rnorm(d*n,mean=0,sd=sd_small),nrow=n,ncol=d)



    # compute the log-densities and the differences
    mean0_vec_true = U %*% mean_vec0
    mean1_vec_true = U %*% mean_vec1

    sigma0_true = U %*% sigma0 %*% t(U) + sd_small^2 * diag(d)
    sigma1_true = U %*% sigma1 %*% t(U) + sd_small^2 * diag(d)

    # set the range of the uniform dist
    k=4
    range_min = mean0_vec_true - 4 * sqrt(diag(sigma0_true))
    range_max = mean1_vec_true + 4 * sqrt(diag(sigma0_true))

    # indices for the elements to be replaced by uniform components
    indices_unif = rbinom(n, size = 1, prob = unif_w)
    n_unif = sum(indices_unif)

    for(j in 1:d){
      data[which(indices_unif==1),j] = runif(n_unif, min=range_min[j], max=range_max[j])
    }

    log_dens_0 = log((1-unif_w) * dmvnorm(data, mean0_vec_true, sigma0_true) +
                       unif_w * prod(range_max - range_min)^(-1))

    log_dens_1 = log((1-unif_w) * dmvnorm(data, mean1_vec_true, sigma1_true) +
                       unif_w * prod(range_max - range_min)^(-1))

    differences_log_dens = log_dens_0 - log_dens_1
  }

  if(scenario == "latent_location_shift"){

    Q_full = randortho(d)

    m = 4
    U = Q_full[, 1:m]

    dim(U)

    mean_vec0 = c(-0.5,0,0,0)
    mean_vec1 = c(0.5,0,0,0)

    sigma0 = diag(c(1,1,1,1)^2)
    sigma1 = diag(c(1,1,1,1)^2)

    data_u = matrix(NA, nrow = n, ncol = m)

    sd_small = 0.1

    mean_unif = c(0,0,0,0)
    sd_unif = 2
    sigma_unif = diag(rep(sd_unif, m)^2)

    for(i in 1:n){

      prob_mix = runif(1)

      if(group_labels[i] == 0){
        if(prob_mix > unif_w){
          u = t(rmvnorm(1, mean_vec0, sigma0))
        }else{
          u = t(rmvnorm(1, mean_unif, sigma_unif))
        }
      }else{
        if(prob_mix > unif_w){
          u = t(rmvnorm(1, mean_vec1, sigma1))
        }else{
          u = t(rmvnorm(1, mean_unif, sigma_unif))
        }
      }
      data_u[i,] = u
    }

    #plot(data_u[which(group_labels == 0),1], data_u[which(group_labels == 0),2])
    #points(data_u[which(group_labels == 1),1], data_u[which(group_labels == 1),2], col = "red")

    data_before_transform = data_u %*% t(U) + matrix(rnorm(d*n,mean=0,sd=sd_small),nrow=n,ncol=d)

    #plot(data[which(group_labels == 0),1], data[which(group_labels == 0),2])
    #points(data[which(group_labels == 1),1], data[which(group_labels == 1),2], col = "red")

    # compute the means and covariances of the d-dim distribution
    mean0_vec_true = U %*% mean_vec0
    mean1_vec_true = U %*% mean_vec1
    mean_unif_vec_true = U %*% mean_unif

    sigma0_true = U %*% sigma0 %*% t(U) + sd_small^2 * diag(d)
    sigma1_true = U %*% sigma1 %*% t(U) + sd_small^2 * diag(d)
    sigma_unif_true = U %*% sigma_unif %*% t(U) + sd_small^2 * diag(d)

    # if necessary, transform to change the marginal distributions
    if(transform){
      # a_beta = 0.5
      # b_beta = 5
      for(j in 1:d){
        data[,j] = qbeta(pnorm(data_before_transform[,j], mean_unif_vec_true[j], sqrt(sigma_unif_true[j,j])),a_beta,b_beta)
      }
    }else{
      data = data_before_transform
    }

    log_dens_0 = log((1-unif_w) * dmvnorm(data_before_transform, mean0_vec_true, sigma0_true) +
                       unif_w * dmvnorm(data_before_transform, mean_unif_vec_true, sigma_unif_true))

    log_dens_1 = log((1-unif_w) * dmvnorm(data_before_transform, mean1_vec_true, sigma1_true) +
                       unif_w * dmvnorm(data_before_transform, mean_unif_vec_true, sigma_unif_true))

    differences_log_dens = log_dens_0 - log_dens_1
  }

  if(scenario == "latent_dispersion_oldest"){

    Q_full = randortho(d)

    m = 4
    U = Q_full[, 1:m]

    dim(U)

    mean_vec0 = c(0,0,0,0)
    mean_vec1 = c(0,0,0,0)

    sigma0 = diag(c(1,1,1,1)^2)
    sigma1 = diag(c(1,1,1,1)^2)
    sigma1[1,2] = 0.8
    sigma1[2,2] = 0.8

    data_eps = matrix(NA, nrow = n, ncol = m)

    sd_small = 0.1

    for(i in 1:n){
      if(group_labels[i] == 0){
        eps = t(rmvnorm(1, mean_vec0, sigma0))
        data_eps[i,] = eps
        data[i,] = U %*% eps + t(rmvnorm(1, sigma = sd_small^2 * diag(d)))
      }else{
        eps = t(rmvnorm(1, mean_vec1, sigma1))
        data_eps[i,] = eps
        data[i,] = U %*% eps + t(rmvnorm(1, sigma = sd_small^2 * diag(d)))
      }
    }

    # compute the log-densities and the differences
    mean0_vec_true = U %*% mean_vec0
    mean1_vec_true = U %*% mean_vec1

    sigma0_true = U %*% sigma0 %*% t(U) + sd_small^2 * diag(d)
    sigma1_true = U %*% sigma1 %*% t(U) + sd_small^2 * diag(d)

    differences_log_dens = dmvnorm(data, mean0_vec_true, sigma0_true,log = TRUE) -
      dmvnorm(data, mean1_vec_true, sigma1_true,log = TRUE)
  }

  if(scenario == "latent_dispersion"){

    Q_full = randortho(d)

    m = 4
    U = Q_full[, 1:m]

    dim(U)

    mean_vec0 = c(0,0,0,0)
    mean_vec1 = c(0,0,0,0)

    sigma0 = diag(c(1,1,1,1)^2)
    sigma1 = diag(c(0.5,1,1,1)^2)
    #sigma1[1,2] = sigma1[2,1] = 0.7

    data_u = matrix(NA, nrow = n, ncol = m)

    sd_small = 0.1

    mean_unif = c(0,0,0,0)
    sd_unif = 2
    sigma_unif = diag(rep(sd_unif, m)^2)

    for(i in 1:n){

      prob_mix = runif(1)

      if(group_labels[i] == 0){
        if(prob_mix > unif_w){
          u = t(rmvnorm(1, mean_vec0, sigma0))
        }else{
          u = t(rmvnorm(1, mean_unif, sigma_unif))
        }
      }else{
        if(prob_mix > unif_w){
          u = t(rmvnorm(1, mean_vec1, sigma1))
        }else{
          u = t(rmvnorm(1, mean_unif, sigma_unif))
        }
      }
      data_u[i,] = u
    }

    #plot(data_u[which(group_labels == 0),1], data_u[which(group_labels == 0),2])
    #points(data_u[which(group_labels == 1),1], data_u[which(group_labels == 1),2], col = "red")

    data_before_transform = data_u %*% t(U) + matrix(rnorm(d*n,mean=0,sd=sd_small),nrow=n,ncol=d)

    #plot(data[which(group_labels == 0),1], data[which(group_labels == 0),2])
    #points(data[which(group_labels == 1),1], data[which(group_labels == 1),2], col = "red")

    # compute the means and covariances of the d-dim distribution
    mean0_vec_true = U %*% mean_vec0
    mean1_vec_true = U %*% mean_vec1
    mean_unif_vec_true = U %*% mean_unif

    sigma0_true = U %*% sigma0 %*% t(U) + sd_small^2 * diag(d)
    sigma1_true = U %*% sigma1 %*% t(U) + sd_small^2 * diag(d)
    sigma_unif_true = U %*% sigma_unif %*% t(U) + sd_small^2 * diag(d)

    # if necessary, transform to change the marginal distributions
    if(transform){
      # a_beta = 0.5
      # b_beta = 5
      for(j in 1:d){
        data[,j] = qbeta(pnorm(data_before_transform[,j], mean_unif_vec_true[j], sqrt(sigma_unif_true[j,j])),a_beta,b_beta)
      }
    }else{
      data = data_before_transform
    }

    log_dens_0 = log((1-unif_w) * dmvnorm(data_before_transform, mean0_vec_true, sigma0_true) +
                       unif_w * dmvnorm(data_before_transform, mean_unif_vec_true, sigma_unif_true))

    log_dens_1 = log((1-unif_w) * dmvnorm(data_before_transform, mean1_vec_true, sigma1_true) +
                       unif_w * dmvnorm(data_before_transform, mean_unif_vec_true, sigma_unif_true))

    differences_log_dens = log_dens_0 - log_dens_1
  }

  out = list("data" = data,
             "data_u" = data_u,
             "group_labels" = group_labels,
             "true_log_w_obs" = differences_log_dens)

  return(out)
}
