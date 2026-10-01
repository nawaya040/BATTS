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
              ivec labels_train,
              double n_min_obs_per_node,
              double n_ratio_per_node,
              bool use_gradient,
              int size_burnin,
              int size_backfitting,
              int thin,
              vec prob_moves,
              double lambda_0,
              double a_prior_omega,
              double b_prior_omega,
              double alpha_tree,
              double beta_tree,
              bool output_BART_ensembles,
              bool quiet
){

  if(thin < 1){
    Rcpp::stop("thin must be a positive integer");
  }

  // BART starts from unsplit trees; split boosting trees can carry an
  // external normalization factor into the posterior updates.
  if(size_backfitting > 0 && max_resol != 0){
    Rcpp::stop("BART requires max_resol = 0 so every initial tree is unsplit");
  }

  class_balancePM my_boosting(
      X,
      group_labels,
      num_trees,
      max_resol,
      learn_rate,
      L_candidates,
      labels_train,
      n_min_obs_per_node,
      n_ratio_per_node,
      use_gradient,
      size_burnin,
      size_backfitting,
      thin,
      prob_moves,
      lambda_0,
      a_prior_omega,
      b_prior_omega,
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
