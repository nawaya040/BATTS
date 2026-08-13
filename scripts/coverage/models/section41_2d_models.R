simulation_2d = function(n0, n1, scenario, n_grid_per_dim){

  library(mvtnorm)

  d = 2

  group_labels = c(rep(0, n0), rep(1, n1))
  n = n0 + n1
  data = matrix(NA, nrow = n, ncol = d)

  differences_log_dens = numeric(n)

  if(scenario == 1){
    # Scenario 1: global difference (from Soriano and Ma; 2017)
    mu = c(0, 0)
    #delta = c(2, 0)
    delta = c(1, 0)
    Sig = matrix(NA, ncol = 2, nrow = 2)
    Sig[1,1] = 2.0
    Sig[1,2] = Sig[2,1] = 0.4
    Sig[2,2] = 1.0

    # simulation

    X0 = rmvnorm(n0, mean=mu, sigma=Sig)
    X1 = rmvnorm(n1, mean=mu+delta, sigma=Sig)

    data = rbind(X0, X1)

    # evaluation of the densities

    log_densities_0 = dmvnorm(data, mean = mu, sigma = Sig, log = T)
    log_densities_1 = dmvnorm(data, mean = mu+delta, sigma = Sig, log = T)

    differences_log_dens = log_densities_0 - log_densities_1
  }

  if(scenario == 2){
    # Scenario 2: local shift
    w = c(1/3, 1/3, 1/3)
    cum_w = cumsum(w)

    means1 = c(-2.5,1,2)
    means2 = c(1,-2,2.5)

    means = rbind(means1, means2)

    sd1 = 0.3
    sd2 = 0.3

    shift = c(-sd1, -0.0)

    Sig = matrix(0, nrow = 2, ncol = 2)
    Sig[1,1] = sd1^2
    Sig[2,2] = sd2^2

    # simulation

    for(i in 1:n){

      group_i = group_labels[i]
      u = runif(1)

      if(u < cum_w[1]){

        data[i,] = rmvnorm(1, mean=means[,1], sigma=Sig)

      }else if(u < cum_w[2]){

        data[i,] = rmvnorm(1, mean=means[,2], sigma=Sig)

      }else{

        if(group_i == 1){

          data[i,] = rmvnorm(1, mean=means[,3] + shift, sigma=Sig)

        }else{

          data[i,] = rmvnorm(1, mean=means[,3], sigma=Sig)

        }

      }
    }

    # evaluate the densities
    log_densities_0 = numeric(n)
    log_densities_1 = numeric(n)

    log_densities_0 = log(w[1] * dmvnorm(data, mean = means[,1], sigma = Sig) +
                            w[2] * dmvnorm(data, mean = means[,2], sigma = Sig) +
                            w[3] * dmvnorm(data, mean = means[,3], sigma = Sig))

    log_densities_1 = log(w[1] * dmvnorm(data, mean = means[,1], sigma = Sig) +
                            w[2] * dmvnorm(data, mean = means[,2], sigma = Sig) +
                            w[3] * dmvnorm(data, mean = means[,3] + shift, sigma = Sig))

    differences_log_dens = log_densities_0 - log_densities_1
  }

  if(scenario == 3){
    # Scenario 3: local dispersion difference
    w = c(1/3, 1/3, 1/3)
    cum_w = cumsum(w)

    means1 = c(-2.5,1,2)
    means2 = c(1,-2,2.5)

    means = rbind(means1, means2)

    sd1 = 0.3
    sd2 = 0.3

    Sig = matrix(0, nrow = 2, ncol = 2)
    Sig[1,1] = sd1^2
    Sig[2,2] = sd2^2

    sd_scale = 0.25

    Sig_another = Sig
    Sig_another[1,1] = Sig[1,1] * sd_scale
    Sig_another[2,2] = Sig[2,2] * sd_scale

    # simulation

    for(i in 1:n){

      group_i = group_labels[i]
      u = runif(1)

      if(u < cum_w[1]){

        data[i,] = rmvnorm(1, mean=means[,1], sigma=Sig)

      }else if(u < cum_w[2]){

        data[i,] = rmvnorm(1, mean=means[,2], sigma=Sig)

      }else{

        if(group_i == 1){

          data[i,] = rmvnorm(1, mean=means[,3], sigma=Sig_another)

        }else{

          data[i,] = rmvnorm(1, mean=means[,3], sigma=Sig)

        }

      }
    }

    log_densities_0 = numeric(n)
    log_densities_1 = numeric(n)

    log_densities_0 = log(w[1] * dmvnorm(data, mean = means[,1], sigma = Sig) +
                            w[2] * dmvnorm(data, mean = means[,2], sigma = Sig) +
                            w[3] * dmvnorm(data, mean = means[,3], sigma = Sig))

    log_densities_1 = log(w[1] * dmvnorm(data, mean = means[,1], sigma = Sig) +
                            w[2] * dmvnorm(data, mean = means[,2], sigma = Sig) +
                            w[3] * dmvnorm(data, mean = means[,3], sigma = Sig_another))

    differences_log_dens = log_densities_0 - log_densities_1
  }

  if(scenario == "global_shift"){
    # Scenario 1: global difference (from Soriano and Ma; 2017)
    mu = c(-0.5, -0.5)
    #delta = c(2, 0)
    delta = c(0.5, 0.5)
    Sig = matrix(NA, ncol = 2, nrow = 2)
    Sig[1,1] = 1
    Sig[1,2] = Sig[2,1] = 0
    Sig[2,2] = 1.0

    k = 4
    range_x = c(mu[1]-k*sqrt(Sig[1,1]), mu[1]+delta[1]+k*sqrt(Sig[1,1]))
    range_y = c(mu[2]-k*sqrt(Sig[2,2]), mu[2]+delta[2]+k*sqrt(Sig[2,2]))

    unif_w = 0.0

    # simulation
    X0 = rmvnorm(n0, mean=mu, sigma=Sig)
    X1 = rmvnorm(n1, mean=mu+delta, sigma=Sig)

    data = rbind(X0, X1)

    # indices for the elements to be replaced by uniform components
    indices_unif = rbinom(n, size = 1, prob = unif_w)
    n_unif = sum(indices_unif)

    data[which(indices_unif==1),] = matrix(c(runif(n_unif, min=range_x[1], max=range_x[2]),
                                            runif(n_unif, min=range_y[1], max=range_y[2])), ncol = 2)

    # evaluation of the densities
    unif_dens = 1 / ((range_x[2]-range_x[1]) * (range_y[2]-range_y[1]))
    log_densities_0 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(data, mean = mu, sigma = Sig, log = F))
    log_densities_1 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(data, mean = mu+delta, sigma = Sig, log = F))

    true_log_densities_0 = log_densities_0
    true_log_densities_1 = log_densities_1

    differences_log_dens = log_densities_0 - log_densities_1

    #Make the grid points
    x.grid = seq(range_x[1],range_x[2],length.out = n_grid_per_dim)
    y.grid = seq(range_y[1],range_y[2],length.out = n_grid_per_dim)
    grid.points = as.matrix(expand.grid(x.grid,y.grid))

    log_densities_0 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(grid.points, mean = mu, sigma = Sig, log = F))
    log_densities_1 = log(unif_w * unif_dens + (1-unif_w) * dmvnorm(grid.points, mean = mu+delta, sigma = Sig, log = F))

    differences_log_dens_grid = log_densities_0 - log_densities_1

  }

  if(scenario == "local_shift"){
    unif_w = 0.0


    arr = array(NA, dim = c(2, 2, 5))
    arr[, , 1]

    p_vec = c(0.2, 0.2, 0.2, 0.2, 0.2)
    delta_vec = c(0.0, 1)

    mu_mat = matrix(NA, nrow=2, ncol=5)
    mu_mat[,1] = c(9.0, 9.9)
    mu_mat[,2] = c(-2.5, 1.4)
    mu_mat[,3] = c(-2.3, -9.7)
    mu_mat[,4] = c(3.4, 5.9)
    mu_mat[,5] = c(5.8, -9.5)

    mu_alt_mat = mu_mat
    dim_with_dif = 5
    mu_alt_mat[,dim_with_dif] = mu_alt_mat[,dim_with_dif] + delta_vec

    vec_to_cov_mat = function(x){
      return(matrix(c(x[1],x[2],x[2],x[3]),nrow=2,ncol=2))
    }


    Sigma_array = array(NA, dim = c(2,2,5))
    Sigma_array[,,1] = vec_to_cov_mat(c(2.9, 0.5, 1.1))
    Sigma_array[,,2] = vec_to_cov_mat(c(1.2, -0.6, 2.8))
    Sigma_array[,,3] = vec_to_cov_mat(c(2.3, -1.0, 1.7))
    Sigma_array[,,4] = vec_to_cov_mat(c(1.1, -0.4, 2.9))
    Sigma_array[,,5] = vec_to_cov_mat(c(3.0, 0.2, 1.0))

    n = n0 + n1

    data0 = matrix(NA, nrow=n0, ncol = 2)
    data1 = matrix(NA, nrow=n1, ncol = 2)

    for(i in 1:n0){
      ind = sample(1:5, 1, prob = p_vec)
      data0[i,] = rmvnorm(1, mu_mat[,ind], Sigma_array[,,ind])
    }

    for(i in 1:n1){
      ind = sample(1:5, 1, prob = p_vec)
      data1[i,] = rmvnorm(1, mu_alt_mat[,ind], Sigma_array[,,ind])
    }

    data = rbind(data0, data1)

    k=4
    range_x = c(mu_mat[1,3]-k*sqrt(Sigma_array[1,1,3]), mu_alt_mat[1,1]+k*sqrt(Sigma_array[1,1,1]))
    range_y = c(mu_mat[2,3]-k*sqrt(Sigma_array[2,2,3]), mu_mat[2,1]+k*sqrt(Sigma_array[2,2,1]))

    indices_unif = rbinom(n, size = 1, prob = unif_w)
    n_unif = sum(indices_unif)

    data[which(indices_unif==1),] = matrix(c(runif(n_unif, min=range_x[1], max=range_x[2]),
                                             runif(n_unif, min=range_y[1], max=range_y[2])), ncol = 2)

    # evaluation of the densities
    unif_dens = 1 / ((range_x[2]-range_x[1]) * (range_y[2]-range_y[1]))

    log_densities_0 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(data, mean = mu_mat[,1], sigma = Sigma_array[,,1], log = F) +
                              p_vec[2] * dmvnorm(data, mean = mu_mat[,2], sigma = Sigma_array[,,2], log = F) +
                              p_vec[3] * dmvnorm(data, mean = mu_mat[,3], sigma = Sigma_array[,,3], log = F) +
                              p_vec[4] * dmvnorm(data, mean = mu_mat[,4], sigma = Sigma_array[,,4], log = F) +
                              p_vec[5] * dmvnorm(data, mean = mu_mat[,5], sigma = Sigma_array[,,5], log = F)
                            )
                          )

    log_densities_1 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(data, mean = mu_alt_mat[,1], sigma = Sigma_array[,,1], log = F) +
                                p_vec[2] * dmvnorm(data, mean = mu_alt_mat[,2], sigma = Sigma_array[,,2], log = F) +
                                p_vec[3] * dmvnorm(data, mean = mu_alt_mat[,3], sigma = Sigma_array[,,3], log = F) +
                                p_vec[4] * dmvnorm(data, mean = mu_alt_mat[,4], sigma = Sigma_array[,,4], log = F) +
                                p_vec[5] * dmvnorm(data, mean = mu_alt_mat[,5], sigma = Sigma_array[,,5], log = F)
                            )
    )

    differences_log_dens = log_densities_0 - log_densities_1

    true_log_densities_0 = log_densities_0
    true_log_densities_1 = log_densities_1

    #Make the grid points
    x.grid = seq(range_x[1],range_x[2],length.out = n_grid_per_dim)
    y.grid = seq(range_y[1],range_y[2],length.out = n_grid_per_dim)
    grid.points = as.matrix(expand.grid(x.grid,y.grid))


    log_densities_0 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(grid.points, mean = mu_mat[,1], sigma = Sigma_array[,,1], log = F) +
                                p_vec[2] * dmvnorm(grid.points, mean = mu_mat[,2], sigma = Sigma_array[,,2], log = F) +
                                p_vec[3] * dmvnorm(grid.points, mean = mu_mat[,3], sigma = Sigma_array[,,3], log = F) +
                                p_vec[4] * dmvnorm(grid.points, mean = mu_mat[,4], sigma = Sigma_array[,,4], log = F) +
                                p_vec[5] * dmvnorm(grid.points, mean = mu_mat[,5], sigma = Sigma_array[,,5], log = F)
                            )
    )

    log_densities_1 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(grid.points, mean = mu_alt_mat[,1], sigma = Sigma_array[,,1], log = F) +
                                p_vec[2] * dmvnorm(grid.points, mean = mu_alt_mat[,2], sigma = Sigma_array[,,2], log = F) +
                                p_vec[3] * dmvnorm(grid.points, mean = mu_alt_mat[,3], sigma = Sigma_array[,,3], log = F) +
                                p_vec[4] * dmvnorm(grid.points, mean = mu_alt_mat[,4], sigma = Sigma_array[,,4], log = F) +
                                p_vec[5] * dmvnorm(grid.points, mean = mu_alt_mat[,5], sigma = Sigma_array[,,5], log = F)
                            )
    )

    differences_log_dens_grid = log_densities_0 - log_densities_1



  }

  if(scenario == "local_dispersion"){
    unif_w = 0.0

    arr = array(NA, dim = c(2, 2, 5))
    arr[, , 1]

    p_vec = c(1/3, 0.0, 1/3, 1/3, 0.0)

    mu_mat = matrix(NA, nrow=2, ncol=5)
    mu_mat[,1] = c(1.9, -7.2)
    mu_mat[,2] = c(-5.7, 5.3)
    mu_mat[,3] = c(-2.3, -1.5)
    mu_mat[,4] = c(7.5, -3.1)
    mu_mat[,5] = c(3.1, 9.5)

    vec_to_cov_mat = function(x){
      return(matrix(c(x[1],x[2],x[2],x[3]),nrow=2,ncol=2))
    }


    Sigma_array = array(NA, dim = c(2,2,5))
    Sigma_array[,,1] = vec_to_cov_mat(c(1.0, -0.4, 0.8)) # changed from the original
    Sigma_array[,,2] = vec_to_cov_mat(c(1.3, 0.7, 2.7))
    Sigma_array[,,3] = vec_to_cov_mat(c(1.0, 0, 3.0))
    Sigma_array[,,4] = vec_to_cov_mat(c(2.9, 0, 1.1))
    Sigma_array[,,5] = vec_to_cov_mat(c(2.4, -0.9, 1.6))

    Sigma_array_alt = Sigma_array
    dim_change = 4
    scale_change_x = 0.6
    scale_change_y = 1.0

    Sigma_array_alt[1,1,dim_change] = scale_change_x^2 * Sigma_array_alt[1,1,dim_change]
    Sigma_array_alt[2,2,dim_change] = scale_change_y^2 * Sigma_array_alt[2,2,dim_change]
    Sigma_array_alt[1,2,dim_change] = scale_change_x * scale_change_y * Sigma_array_alt[1,2,dim_change]
    Sigma_array_alt[2,1,dim_change] = scale_change_x * scale_change_y * Sigma_array_alt[2,1,dim_change]

    # swap two groups
    Sigma_temp = Sigma_array
    Sigma_array = Sigma_array_alt
    Sigma_array_alt = Sigma_temp

    #for(j in 1:5){
    #  print(mu_mat[,j] - 4 * sqrt(c(Sigma_array[1,1,j], Sigma_array[2,2,j])))
    #}

    n = n0 + n1

    data0 = matrix(NA, nrow=n0, ncol = 2)
    data1 = matrix(NA, nrow=n1, ncol = 2)

    for(i in 1:n0){
      ind = sample(1:5, 1, prob = p_vec)
      data0[i,] = rmvnorm(1, mu_mat[,ind], Sigma_array[,,ind])
    }

    for(i in 1:n1){
      ind = sample(1:5, 1, prob = p_vec)
      data1[i,] = rmvnorm(1, mu_mat[,ind], Sigma_array_alt[,,ind])
    }

    data = rbind(data0, data1)

    k=4
    range_x = c(mu_mat[1,2]-k*sqrt(Sigma_array[1,1,2]), mu_mat[1,4]+k*sqrt(Sigma_array[1,1,4]))
    range_y = c(mu_mat[2,1]-k*sqrt(Sigma_array[2,2,1]), mu_mat[2,2]+k*sqrt(Sigma_array[2,2,2]))

    indices_unif = rbinom(n, size = 1, prob = unif_w)
    n_unif = sum(indices_unif)

    data[which(indices_unif==1),] = matrix(c(runif(n_unif, min=range_x[1], max=range_x[2]),
                                             runif(n_unif, min=range_y[1], max=range_y[2])), ncol = 2)

    # evaluation of the densities
    unif_dens = 1 / ((range_x[2]-range_x[1]) * (range_y[2]-range_y[1]))

    log_densities_0 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(data, mean = mu_mat[,1], sigma = Sigma_array[,,1], log = F) +
                                p_vec[2] * dmvnorm(data, mean = mu_mat[,2], sigma = Sigma_array[,,2], log = F) +
                                p_vec[3] * dmvnorm(data, mean = mu_mat[,3], sigma = Sigma_array[,,3], log = F) +
                                p_vec[4] * dmvnorm(data, mean = mu_mat[,4], sigma = Sigma_array[,,4], log = F) +
                                p_vec[5] * dmvnorm(data, mean = mu_mat[,5], sigma = Sigma_array[,,5], log = F)
                            )
    )

    log_densities_1 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(data, mean = mu_mat[,1], sigma = Sigma_array_alt[,,1], log = F) +
                                p_vec[2] * dmvnorm(data, mean = mu_mat[,2], sigma = Sigma_array_alt[,,2], log = F) +
                                p_vec[3] * dmvnorm(data, mean = mu_mat[,3], sigma = Sigma_array_alt[,,3], log = F) +
                                p_vec[4] * dmvnorm(data, mean = mu_mat[,4], sigma = Sigma_array_alt[,,4], log = F) +
                                p_vec[5] * dmvnorm(data, mean = mu_mat[,5], sigma = Sigma_array_alt[,,5], log = F)
                            )
    )

    differences_log_dens = log_densities_0 - log_densities_1

    true_log_densities_0 = log_densities_0
    true_log_densities_1 = log_densities_1

    #Make the grid points
    x.grid = seq(range_x[1],range_x[2],length.out = n_grid_per_dim)
    y.grid = seq(range_y[1],range_y[2],length.out = n_grid_per_dim)
    grid.points = as.matrix(expand.grid(x.grid,y.grid))


    log_densities_0 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(grid.points, mean = mu_mat[,1], sigma = Sigma_array[,,1], log = F) +
                                p_vec[2] * dmvnorm(grid.points, mean = mu_mat[,2], sigma = Sigma_array[,,2], log = F) +
                                p_vec[3] * dmvnorm(grid.points, mean = mu_mat[,3], sigma = Sigma_array[,,3], log = F) +
                                p_vec[4] * dmvnorm(grid.points, mean = mu_mat[,4], sigma = Sigma_array[,,4], log = F) +
                                p_vec[5] * dmvnorm(grid.points, mean = mu_mat[,5], sigma = Sigma_array[,,5], log = F)
                            )
    )

    log_densities_1 = log(unif_w * unif_dens +
                            (1-unif_w) * (
                              p_vec[1] * dmvnorm(grid.points, mean = mu_mat[,1], sigma = Sigma_array_alt[,,1], log = F) +
                                p_vec[2] * dmvnorm(grid.points, mean = mu_mat[,2], sigma = Sigma_array_alt[,,2], log = F) +
                                p_vec[3] * dmvnorm(grid.points, mean = mu_mat[,3], sigma = Sigma_array_alt[,,3], log = F) +
                                p_vec[4] * dmvnorm(grid.points, mean = mu_mat[,4], sigma = Sigma_array_alt[,,4], log = F) +
                                p_vec[5] * dmvnorm(grid.points, mean = mu_mat[,5], sigma = Sigma_array_alt[,,5], log = F)
                            )
    )

    differences_log_dens_grid = log_densities_0 - log_densities_1



  }

  out = list("data" = data,
             "group_labels" = group_labels,
             "true_log_w_obs" = differences_log_dens,
             "true_log_w_grid" = differences_log_dens_grid,
             "grid_points" = grid.points,
             "true_log_densities_0"  = true_log_densities_0,
             "true_log_densities_1"  = true_log_densities_1)

  return(out)
}
