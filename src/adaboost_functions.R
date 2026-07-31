estimate_balancing_weight = function(data,
                     group_labels,
                     num_trees = 100,
                     K_CV = 0 ,
                     max_resol = 4,
                     learn_rate = 0.01,
                     n_cut_points = 100,
                     n_min_obs_per_node = 1,
                     margin_scale = 0.1,
                     use_gradient = FALSE,
                     size_burnin = 0,
                     size_backfitting = 0,
                     lambda_prior_parameters = c(1,1),
                     omega_prior_parameters = c(1,1),
                     tree_priors = c(0.95,2.0),
                     output_BART_ensembles = FALSE,
                     quiet = FALSE
                     ){

  #Re-scale the data
  d = ncol(data)
  min_max_values = matrix(NA, nrow = d, ncol = 2)

  if(margin_scale >= 0){
    for(j in 1:d){
      min_j = min(data[,j])
      max_j = max(data[,j])

      margin_size = (max_j - min_j) * margin_scale
      min_j_new = min_j - margin_size
      max_j_new = max_j + margin_size

      data[,j] = (data[,j] - min_j_new) / (max_j_new - min_j_new)
      min_max_values[j,1] = min_j_new
      min_max_values[j,2] = max_j_new
    }
  }else{
    min_max_values[,1] = 0
    min_max_values[,2] = 1
  }


  #obtain the information of the data
  n0 = sum(group_labels == 0)
  n1 = sum(group_labels == 1)

  data_info = list("n0" = n0,
                   "n1" = n1,
                   "d" = d,
                   "min_max_values" = min_max_values,
                   "training_data" = data)

  # optimize the number of trees if the option is yes
  if(K_CV > 0){

    labels_CV = c(sample(rep(1:K_CV, length.out = sum(group_labels==0))), sample(rep(1:K_CV, length.out = sum(group_labels==1))))
    loss_CV_store = matrix(NA, nrow = K_CV, ncol = num_trees)

    for(k in 1:K_CV){

      labels_train = numeric(length(group_labels))
      labels_train[which(labels_CV != k)] = 1

      # boosting
      out_CV = run_adaboost(data,
                         group_labels,
                         num_trees,
                         max_resol,
                         learn_rate,
                         n_cut_points,
                         labels_train,
                         n_min_obs_per_node,
                         use_gradient,
                         0,
                         0,
                         lambda_prior_parameters[1],
                         lambda_prior_parameters[2],
                         omega_prior_parameters[1],
                         omega_prior_parameters[2],
                         tree_priors[1],
                         tree_priors[2],
                         FALSE,
                         quiet
      )

      loss_CV_store[k,] = out_CV$loss_curve
    }

    num_trees_opt = which.min(colMeans(loss_CV_store))

  }

  #Run boosting
  out = run_adaboost(data,
                 group_labels,
                 num_trees_opt,
                 max_resol,
                 learn_rate,
                 n_cut_points,
                 rep(1, length(group_labels)),
                 n_min_obs_per_node,
                 use_gradient,
                 size_burnin,
                 size_backfitting,
                 lambda_prior_parameters[1],
                 lambda_prior_parameters[2],
                 omega_prior_parameters[1],
                 omega_prior_parameters[2],
                 tree_priors[1],
                 tree_priors[2],
                 output_BART_ensembles,
                 quiet
  )

  # re-scalte the values of the residuals
  for(j in 1:d){
    min_j_new = min_max_values[j,1]
    max_j_new = min_max_values[j,2]

    out$residuals_current[,j] = min_j_new + out$residuals_current[,j] * (max_j_new - min_j_new)
  }

  #out = c(out, data_info = list(data_info))
  out$data_info = data_info

  if(K_CV > 0){
    out$loss_CV_store = loss_CV_store
  }

  out$Omega = min_max_values

  return(out)
}

eval_balance_weight = function(list_boosting, eval_points, BART_result = FALSE){

  data_info = list_boosting$data_info

  if(length(list_boosting$tree_list) == 0){

    stop("The tree list is empty")

  }else{
    #re-scale the input data
    min_max_values = data_info$min_max_values
    d = data_info$d

    for(j in 1:d){
      min_j_new = min_max_values[j,1]
      max_j_new = min_max_values[j,2]

      if(sum(eval_points[,j] < min_j_new) > 0 || sum(eval_points[,j] > max_j_new) > 0){
        stop("Some points are outside of the sample space")
      }

      eval_points[,j] = (eval_points[,j] - min_j_new) / (max_j_new - min_j_new)
    }

    out = list()

    out_temp = evaluate_balance_weight_boosting(list_boosting$tree_list, eval_points)
    out$balancing_weight_boosting = out_temp$balance_current

    if(BART_result){
      out_temp = evaluate_balance_weight_BART(list_boosting$forest_list, eval_points)
      out$balancing_weight_BART = out_temp$balance_store
    }
  }

  return(out)

}

eval_balance_weight_boosting = function(list_boosting, eval_points){

  data_info = list_boosting$data_info

  if(length(list_boosting$tree_list) == 0){

    out = list()

    out$balance_weights = rep(1, nrow(eval_points))

  }else{
    #re-scale the input data
    min_max_values = data_info$min_max_values
    d = data_info$d

    for(j in 1:d){
      min_j_new = min_max_values[j,1]
      max_j_new = min_max_values[j,2]

      eval_points[,j] = (eval_points[,j] - min_j_new) / (max_j_new - min_j_new)
    }

    out = evaluate_balance_weight_boosting(list_boosting$tree_list, eval_points)
  }

  return(out)
}

eval_balance_weight_BART = function(list_boosting, eval_points){

  data_info = list_boosting$data_info

  if(length(list_boosting$tree_list) == 0){

    out = list()

    out$balance_weights = rep(1, nrow(eval_points))

  }else{
    #re-scale the input data
    min_max_values = data_info$min_max_values
    d = data_info$d

    for(j in 1:d){
      min_j_new = min_max_values[j,1]
      max_j_new = min_max_values[j,2]

      eval_points[,j] = (eval_points[,j] - min_j_new) / (max_j_new - min_j_new)
    }

    out = evaluate_balance_weight_BART(list_boosting$forest_list, eval_points)
  }

  return(out)
}
