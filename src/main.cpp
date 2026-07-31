// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "class_balancePM.h"
#include "helpers.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
List run_adaboost(mat X,
              ivec group_labels,
              int num_trees,
              int max_resol,
              double learn_rate,
              vec L_candidates,
              double alpha_cutpoint,
              ivec labels_train,
              double n_min_obs_per_node,
              double n_ratio_per_node,
              bool use_gradient,
              int size_burnin,
              int size_backfitting,
              int thin,
              vec prob_moves,
              double lambda_0,
              double a_prior_lambda,
              double b_prior_lambda,
              double a_prior_omega,
              double b_prior_omega,
              bool update_lambda,
              double alpha_tree,
              double beta_tree,
              bool output_BART_ensembles,
              bool quiet
){

  class_balancePM my_boosting(
      X,
      group_labels,
      num_trees,
      max_resol,
      learn_rate,
      L_candidates,
      alpha_cutpoint,
      labels_train,
      n_min_obs_per_node,
      n_ratio_per_node,
      use_gradient,
      size_burnin,
      size_backfitting,
      thin,
      prob_moves,
      lambda_0,
      a_prior_lambda,
      b_prior_lambda,
      a_prior_omega,
      b_prior_omega,
      update_lambda,
      alpha_tree,
      beta_tree,
      output_BART_ensembles,
      quiet
  );

  my_boosting.do_boosting();

  if(size_backfitting > 0){
    my_boosting.backfitting();
  }

  List out = my_boosting.output();

  return out;

}
