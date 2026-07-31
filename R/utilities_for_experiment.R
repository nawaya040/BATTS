ada_log_weight = function(data,
                          group_labels,
                          grid.points = NULL,
                          K_CV = 5,
                          num_trees_max = 500,
                          num_trees_min = 100,
                          maxdepth = 4,
                          learn_rate = 0.01,
                          test_error = TRUE
                          ){

  n0 = sum(group_labels == 0)
  n1 = sum(group_labels == 1)

  d = ncol(data)

  labels_X = paste("X", 1:d, sep = "")

  df = data.frame(y = factor(group_labels), data = data)
  colnames(df)[-1] = labels_X

  num_tree_opt_ada = num_trees_max

  control = rpart.control(maxdepth = maxdepth,cp = -1, minsplit = 0)

  if(K_CV > 0){

    labels_CV = c(sample(rep(1:K_CV, length.out = sum(group_labels==0))),
                  sample(rep(1:K_CV, length.out = sum(group_labels==1))))
    loss_CV_store_test = matrix(NA, nrow = K_CV, ncol = num_trees_max)
    loss_CV_store_kappa = matrix(NA, nrow = K_CV, ncol = num_trees_max)

    for(k in 1:K_CV){

      labels_train = numeric(length(group_labels))
      labels_train[which(labels_CV != k)] = 1

      df_train = df[which(labels_train==1),]
      df_test = df[which(labels_train==0),]

      # run adaboost
      model_ada = ada(y ~ ., data = df_train, type = "real",
                      control = control,iter=num_trees_max, nu=learn_rate, bag.frac=0.5,
                      test.x = df_test[, -1, drop = FALSE], test.y = df_test$y)

      loss_CV_store_test[k,] = model_ada$model$errs[,"test.err"]
      loss_CV_store_kappa[k,] = model_ada$model$errs[,"test.kap"]
    }

    #plot(colMeans(loss_CV_store), type = "l")
    if(test_error){
      num_tree_opt_ada = max(which.min(colMeans(loss_CV_store_test)), num_trees_min)
    }else{
      num_tree_opt_ada = max(which.min(colMeans(loss_CV_store_kappa)), num_trees_min)
    }
  }


  model = ada(y ~ ., data = df, type = "real",
              control = control,iter=num_tree_opt_ada, nu=0.01, bag.frac=0.5)

  # evaluation on the observed data sets
  prob_pred_data = predict(model, df, type = "prob")
  log_w_hat_ada_data = log(prob_pred_data[,1] / prob_pred_data[,2] * n1 / n0)

  # evaluation on the grid points
  if(!is.null(grid.points)){
    df_grid = data.frame(y = factor(rep(0, nrow(matrix(grid.points)))), grid.points)
    colnames(df_grid)[-1] = labels_X

    prob_pred_grid = predict(model, df_grid, type = "prob")
    log_w_hat_ada_grid = log(prob_pred_grid[,1] / prob_pred_grid[,2] * n1 / n0)
  }

  if(!is.null(grid.points)){
    out = list("log_w_hat_ada_data" = log_w_hat_ada_data,
               "log_w_hat_ada_grid" = log_w_hat_ada_grid,
               "prob_pred_data" = prob_pred_data,
               "prob_pred_grid" = prob_pred_grid,
               # "loss_CV_store_test" = loss_CV_store_test,
               # "loss_CV_store_kappa" = loss_CV_store_kappa,
               "num_trees_selected" = num_tree_opt_ada)
  }else{
    out = list("log_w_hat_ada_data" = log_w_hat_ada_data,
               "num_trees_selected" = num_tree_opt_ada,
               "prob_pred_data" = prob_pred_data
               # "loss_CV_store_test" = loss_CV_store_test,
               # "loss_CV_store_kappa" = loss_CV_store_kappa
               )
  }

}


