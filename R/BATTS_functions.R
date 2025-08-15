#' @export
boots = function(data,
                     group_labels,
                     num_trees_max = 100,
                     K_CV = 0 ,
                     max_resol = 4,
                     learn_rate = 0.01,
                     n_bins = 32,
                     alpha_cutpoint = 1,
                     n_min_obs_per_node = 1,
                     n_ratio_per_node = 1e-100,
                     margin_scale = 0.1,
                     use_gradient = FALSE,
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

  num_trees_opt = num_trees_max

  # make candidates for the cut points
  L_candidates_unif = seq(1/n_bins, 1 - 1/n_bins, by = 1/n_bins)
  y = L_candidates_unif / (1-L_candidates_unif)
  L_candidates = y^(1/alpha_cutpoint) / (1 + y^(1/alpha_cutpoint))

  # inputs for the Bayes method
  # not used, only for avoiding errors
  size_burnin = 0
  size_backfitting = 0
  thin = 1
  prob_moves = c(1/4, 1/4, 1/2)
  lambda_0 = 10
  lambda_prior_parameters = c(1,1)
  omega_prior_parameters = c(1,1)
  update_lambda = FALSE
  tree_priors = c(0.95,2.0)
  output_BART_ensembles = FALSE

  # optimize the number of trees if the option is yes
  if(K_CV > 0){

    labels_CV = c(sample(rep(1:K_CV, length.out = sum(group_labels==0))), sample(rep(1:K_CV, length.out = sum(group_labels==1))))
    loss_CV_store = matrix(NA, nrow = K_CV, ncol = num_trees_max)

    for(k in 1:K_CV){

      print(paste("CV :", k, "/", K_CV), sep = "")

      labels_train = numeric(length(group_labels))
      labels_train[which(labels_CV != k)] = 1

      # boosting
      out_CV = run_adaboost(data,
                         group_labels,
                         num_trees_max,
                         max_resol,
                         learn_rate,
                         L_candidates,
                         alpha_cutpoint,
                         labels_train,
                         n_min_obs_per_node,
                         n_ratio_per_node,
                         use_gradient,
                         0,
                         0,
                         1,
                         prob_moves,
                         lambda_0,
                         lambda_prior_parameters[1],
                         lambda_prior_parameters[2],
                         omega_prior_parameters[1],
                         omega_prior_parameters[2],
                         update_lambda,
                         tree_priors[1],
                         tree_priors[2],
                         FALSE,
                         TRUE
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
                 L_candidates,
                 alpha_cutpoint,
                 rep(1, length(group_labels)),
                 n_min_obs_per_node,
                 n_ratio_per_node,
                 use_gradient,
                 size_burnin,
                 size_backfitting,
                 thin,
                 prob_moves,
                 lambda_0,
                 lambda_prior_parameters[1],
                 lambda_prior_parameters[2],
                 omega_prior_parameters[1],
                 omega_prior_parameters[2],
                 update_lambda,
                 tree_priors[1],
                 tree_priors[2],
                 output_BART_ensembles,
                 quiet
  )

  out$data_info = data_info

  if(K_CV > 0){
    out$loss_CV_store = loss_CV_store
  }

  out$Omega = min_max_values

  return(out)
}

#' @export
batts = function(data,
                                     group_labels,
                                     num_trees = 100,
                                     max_resol = 0,
                                     learn_rate = 0.01,
                                     n_bins = 100,
                                     alpha_cutpoint = 1,
                                     n_min_obs_per_node = 1,
                                     n_ratio_per_node = 1e-100,
                                     margin_scale = 0.1,
                                     use_gradient = FALSE,
                                     size_burnin = NULL,
                                     size_backfitting = 0,
                                     thin = 1,
                                     prob_moves = c(1/3,1/3,1/3),
                                     lambda_0 = 5,
                                     lambda_prior_parameters = c(1,1),
                                     omega_prior_parameters = c(1,1),
                                     update_lambda = FALSE,
                                     tree_priors = c(0.95,2.0),
                                     output_BART_ensembles = FALSE,
                                     quiet = FALSE
){

  if(is.null(size_burnin)){
    size_burnin = size_backfitting / 2
  }

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

  L_candidates_unif = seq(1/n_bins, 1 - 1/n_bins, by = 1/n_bins)
  y = L_candidates_unif / (1-L_candidates_unif)
  L_candidates = y^(1/alpha_cutpoint) / (1 + y^(1/alpha_cutpoint))

  #Run the preliminary boosting and the back-fitting method
  out = run_adaboost(data,
                     group_labels,
                     num_trees,
                     max_resol,
                     learn_rate,
                     L_candidates,
                     alpha_cutpoint,
                     rep(1, length(group_labels)),
                     n_min_obs_per_node,
                     n_ratio_per_node,
                     use_gradient,
                     size_burnin,
                     size_backfitting,
                     thin,
                     prob_moves,
                     lambda_0,
                     lambda_prior_parameters[1],
                     lambda_prior_parameters[2],
                     omega_prior_parameters[1],
                     omega_prior_parameters[2],
                     update_lambda,
                     tree_priors[1],
                     tree_priors[2],
                     output_BART_ensembles,
                     quiet
  )

  out$data_info = data_info
  out$Omega = min_max_values

  return(out)
}

#' @export
eval_balance_weight = function(list_result, eval_points, is_Bayes = FALSE){

  data_info = list_result$data_info

  if(length(list_result$tree_list) == 0){

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

    out_temp = evaluate_balance_weight_boosting(list_result$tree_list, eval_points)
    out$balancing_weight_boosting = out_temp$balance_current

    if(is_Bayes){
      out_temp = evaluate_balance_weight_BART(list_result$forest_list, eval_points)
      out$balancing_weight_BART = out_temp$balance_store
    }
  }

  return(out)

}

