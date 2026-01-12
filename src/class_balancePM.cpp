#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <bits/stdc++.h>
#include "class_balancePM.h"
#include "helpers.h"

using namespace Rcpp;
using namespace arma;
using namespace std;


#define INDEX_ZERO 0
#define NUM_MOVES 2

#define LARGE_NUMBER 1e+100
#define SMALL_NUMBER 1e-100

class_balancePM::class_balancePM(
  mat X,
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
):
  X(X),
  group_labels(group_labels),
  num_trees(num_trees),
  max_resol(max_resol),
  learn_rate(learn_rate),
  L_candidates(L_candidates),
  alpha_cutpoint(alpha_cutpoint),
  labels_train(labels_train),
  n_min_obs_per_node(n_min_obs_per_node),
  n_ratio_per_node(n_ratio_per_node),
  use_gradient(use_gradient),
  size_burnin(size_burnin),
  size_backfitting(size_backfitting),
  thin(thin),
  prob_moves(prob_moves),
  lambda_0(lambda_0),
  a_prior_lambda(a_prior_lambda),
  b_prior_lambda(b_prior_lambda),
  a_prior_omega(a_prior_omega),
  b_prior_omega(b_prior_omega),
  update_lambda(update_lambda),
  alpha_tree(alpha_tree),
  beta_tree(beta_tree),
  output_BART_ensembles(output_BART_ensembles),
  quiet(quiet)
{
  init();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Initialization
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

void class_balancePM::init(){ //Initialization

  //Input the basic information
  n = X.n_rows;
  d = X.n_cols;

  n_cut_points = L_candidates.n_rows;
  n_cells = n_cut_points + 1;

  group_labels_double = zeros(n);
  for(int i=0; i<n; i++){
    group_labels_double(i) = (double) group_labels(i);
  }

  uvec indices_0_u = find(group_labels == 0);
  uvec indices_1_u = find(group_labels == 1);

  indices_0 = arma::conv_to<vector<int>>::from(indices_0_u);
  indices_1 = arma::conv_to<vector<int>>::from(indices_1_u);

  n_vec = zeros<ivec>(2);

  n_vec(1) = sum(group_labels);
  n_vec(0) = n - sum(group_labels);

  // vector to store the current balancing weights
  balance_current = zeros(n);
  balance_inv_current = zeros(n);

  // matrix to store the estimated balancing weights
  // (requires a lot of memory, should be optimized later)
  balance_store_boosting = zeros(n, num_trees);

  // vector to store the current gradients (necessary if we use the gradient boosting)
  gradient_current = zeros(n);

  // subsample the data every iteration
  size_subsample = sum(labels_train);

  // check which observations are in the training or test (validation) set
  indices_used = find(labels_train == 1);
  indices_not_used = find(labels_train == 0);

  // compute # observations in each set
  n0_learn = 0;
  n1_learn = 0;

  n0_validation = 0;
  n1_validation = 0;

  for(int i=0;i<n;i++){
    if(group_labels(i)==0){
      if(labels_train(i)==0){
        n0_validation += 1;
      }else{
        n0_learn += 1;
      }
    }else{
      if(labels_train(i)==0){
        n1_validation += 1;
      }else{
        n1_learn += 1;
      }
    }
  }

  // need to monitor the loss (in the preparation step
  loss_curve = zeros(num_trees);

  // need to store the product of c (normalizing constant) given in each step
  prod_c = 1.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//tree functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

Node* class_balancePM::get_root_node(){

  Node* new_node = new Node;

  new_node->depth = 0;

  new_node->node_id = 1;

  new_node->left_points = zeros<vec>(d);
  new_node->right_points = ones<vec>(d);

  new_node->dim_selected = 0; //Not registered
  new_node->location = 0; //Not registered
  new_node->partition_point = 0; //Not registered

  new_node->beta = 1.0;

  new_node->parent = nullptr;
  new_node->left = nullptr;
  new_node->right = nullptr;

  new_node->loss = 0.0; //Not registered

  for(int i=0; i<size_subsample; i++){
    new_node->indices.push_back(indices_used(i));
  }

  return new_node;
}

Node* class_balancePM::get_new_node(Node* parent, bool this_is_left, int dim_selected, double location){
  //To the parent node, input the information on how this node is split
  //(Note: this step is redundant since we basically do the same thing twice.
  //       But the computation cost to create a new node can be ignored, so the problem does not need to be fixed immediately)

  //Input the dimension
  parent->dim_selected = dim_selected;
  //Input the partition point
  parent->location = location;

  double left = parent->left_points(dim_selected);
  double right = parent->right_points(dim_selected);

  parent->partition_point = left + location * (right - left);

  //Make a new child node
  Node* new_node = new Node;
  new_node->depth = parent->depth+1;

  new_node->left_points = parent->left_points;
  new_node->right_points = parent->right_points;

  if(this_is_left){
    new_node->node_id = 2 * parent->node_id;
    new_node->right_points(dim_selected) = parent->partition_point;
  }else{
    new_node->node_id = 2 * parent->node_id + 1;
    new_node->left_points(dim_selected) = parent->partition_point;
  }

  new_node->dim_selected = 0; //Not registered
  new_node->location = 0; //Not registered
  new_node->partition_point = 0; //Not registered

  new_node->beta = 1.0;

  new_node->parent = parent;
  new_node->left = nullptr;
  new_node->right = nullptr;

  new_node->loss = 0.0; //Not registered

  return new_node;
}


void class_balancePM::add_children(Node* node, int dim_selected, double location){
  node->left = get_new_node(node, true, dim_selected, location);
  node->right = get_new_node(node, false, dim_selected, location);
}

void class_balancePM::count_total_nodes( Node* node, int& count){

  //Make a stack for nodes
  std::stack<Node*> stack_tree;

  //current node
  Node* curr = node;

  while(curr != nullptr || stack_tree.empty() == false){

    while(curr != nullptr){
      stack_tree.push(curr);
      curr = curr->left;
    }

    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();

    curr = curr->right;

    ++count;
  }

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//boosting functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


void class_balancePM::do_boosting(){

  if(!quiet && size_backfitting == 0){
    Rcout << "boosting in progress..." << "\n";
  }

  //Initialize the current residuals here
  residuals_current = X.t(); //NOTICE: transpose -> d x n matrix
  balance_current.fill(1.0); //initial importance weight: all 1
  balance_inv_current.fill(1.0); //initial importance weight: all 1

  for(index_tree=0; index_tree < num_trees; index_tree++){

    //compute the sum of the balancing weights for the two groups
    //they are necessary to obtain the beta
    //note: need to focus on the observations used in constructing the tree
    sum_balance_inv_0 = 0.0;
    sum_balance_1 = 0.0;

    for(int i=0; i<size_subsample; i++){
      uword index = indices_used(i);

      if(group_labels(index) == 0){
        sum_balance_inv_0 += balance_inv_current(index);
      }else{
        sum_balance_1 += balance_current(index);
      }
    }

    // if we use the gradient-based method, need to calculate the current gradients
    // (we only need to calculate the gradients for the training data set)
    if(use_gradient){

    gradient_current.fill(0.0);

      for(int i=0; i<size_subsample; i++){
        uword index = indices_used(i);

        // multiply n to control the scale (we don't in this version)
        if(group_labels(index) == 0){;
          gradient_current(index) = - balance_inv_current(index) / (double) n_vec(0) ;
        }else{
          gradient_current(index) = balance_current(index) / (double) n_vec(1);
        }
      }
    }

    //construct a new tree
    Node* root = get_root_node();

    //before constructing a new tree,
    //input the initial value of the loss we use to split nodes
    if(!use_gradient){
      root->loss = 1.0;
    }else{
      root->loss = sum(gradient_current % gradient_current) / (double) size_subsample
                        - pow(sum(gradient_current) / (double) size_subsample, 2.0);
    }


    //construct a new tree
    construct_tree(root);

    //input the information about each obs is included in which leaf node
    if(size_backfitting > 0){
      input_indices_for_leaves(root);
    }

    //update the balancing weights
    update_balancing_weights(root);

    // normalize the balancing weights
    // to this end, we need to calculate the sum of the balancing weights again

    //note: in this version, we do the normalization only for the Hellinger algorithm

    sum_balance_inv_0 = 0.0;
    sum_balance_1 = 0.0;

    for(int i=0; i<size_subsample; i++){
      uword index = indices_used(i);

      if(group_labels(index) == 0){
        sum_balance_inv_0 += balance_inv_current(index);
      }else{
        sum_balance_1 += balance_current(index);
      }
    }

    if(!use_gradient){
      double c_normalize = (sum_balance_inv_0 / (double) n0_learn) /
        (sum_balance_1 / (double) n1_learn);

      prod_c = prod_c * c_normalize;

      balance_current = c_normalize * balance_current;
      balance_inv_current = 1.0 / c_normalize * balance_inv_current;
    }

    // store the current weight
    balance_store_boosting.col(index_tree) = balance_current;

    // loss given by the validation set (after constructing the tree)
    double loss_temp0 = 0.0;
    double loss_temp1 = 0.0;

    double loss_new = 0.0;

    if(n0_validation > 0 || n1_validation > 0){
      for(int i=0;i<n-size_subsample;i++){

        int index = indices_not_used(i);

        if(group_labels(index) == 0){
          loss_temp0 += balance_inv_current(index);
        }else{
          loss_temp1 += balance_current(index);
        }

      }

      if(n0_validation > 0){
        loss_new += loss_temp0 / (double) n0_validation;
      }

      if(n1_validation > 0){
        loss_new += loss_temp1 / (double) n1_validation;
      }
    }

    loss_curve(index_tree) = loss_new;

    //count # nodes in the current tree (just for debugging)
    int count = 0;
    count_total_nodes(root, count);
    n_nodes.push_back(count);

    //summarize the information of the current tree in the list
    d_store.clear();
    l_store.clear();

    theta_store.clear();
    beta_store.clear();

    record_tree(root);

    List list_curr_tree = Rcpp::List::create(Rcpp::Named("d") = d_store,
                                             Rcpp::Named("l") = l_store,
                                             Rcpp::Named("beta") = beta_store
    );

    tree_list.push_back(list_curr_tree);


    //store the information of the current tree if we will do the backfitting
    //if not, clear the data of the current tree to save the memory
    if(size_backfitting > 0){
      root_nodes.push_back(root);
    }else{
      clear_node(root);
    }

  }

  // store the final result
  balance_boosting = balance_current;

  // prepare for the back-fitting if necessary
  if(size_backfitting > 0){
    num_trees_generated = num_trees;
    num_trees_generated_double = (double) num_trees_generated;
  }

}


void class_balancePM::construct_tree(Node* root){
  //Make a stack for nodes
  std::stack<Node*> stack_tree;

  //current node = root node
  Node* curr = root;

  while(curr != nullptr || stack_tree.empty() == false){

    while(curr != nullptr){
      stack_tree.push(curr);

      //split the node here
      bool is_split = split_node(curr);

      if(!is_split){
        //This is a terminal node
        //compute the value of beta_A
        compute_beta(curr);
      }

    //don't forget to delete the indices!
      clear_indices(curr);

      curr = curr->left;
    }

    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();

    curr = curr->right;
  }

}


bool class_balancePM::split_node(Node* node){

  //output of this function
  //default is false
  bool is_split = false;

  //input the information of the indices
  vector<int> indices_current = node->indices;
  int n_total_A = indices_current.size();

  //if the node is too deep, stop splitting
  //note: we don't need to check the number of residuals included
  //      because in this case the improvement score is always less than the current one

  if(node->depth < max_resol && n_total_A > n_min_obs_per_node){

    //get more information
    mat residuals_A = get_submat_double(residuals_current, indices_current, n_total_A);
    ivec group_labels_A = get_subvector_int(group_labels, indices_current, n_total_A);

    vec left_points_A = node->left_points;
    vec right_points_A = node->right_points;

    // calculate the loss given for the possible splitting rules
    mat loss_left_matrix(d, n_cut_points);
    mat loss_right_matrix(d, n_cut_points);

    mat loss_matrix(d, n_cut_points);

    mat count_0_mat = zeros(d, n_cells);
    mat count_1_mat = zeros(d, n_cells);

    if(!use_gradient){

      vec balance_A = get_subvector_double(balance_current, indices_current, n_total_A);
      vec balance_inv_A = get_subvector_double(balance_inv_current, indices_current, n_total_A);

      // compute the effective size given by the weighted measures for the two groups
      mat sum_inv_0_mat = zeros(d, n_cells);
      mat sum_1_mat = zeros(d, n_cells);

      for(int j=0;j<d; j++){

        double left = left_points_A(j);
        double right = right_points_A(j);

        for(int i=0;i<n_total_A;i++){

          int index_cell = find_cell_x_is_in(residuals_A(j, i), left, right);

          if(group_labels_A(i) == 0){
            sum_inv_0_mat(j,index_cell) = sum_inv_0_mat(j,index_cell) + balance_inv_A(i);
            count_0_mat(j,index_cell) = count_0_mat(j,index_cell) + 1.0;
          }else{
            sum_1_mat(j,index_cell) = sum_1_mat(j,index_cell) + balance_A(i);
            count_1_mat(j,index_cell) = count_1_mat(j,index_cell) + 1.0;
          }

        }

      }

      // calculate the loss
      for(int j=0;j<d; j++){

        vec sum_inv_0_j = sum_inv_0_mat.row(j).t();
        vec sum_1_j = sum_1_mat.row(j).t();

        vec count_0_j = count_0_mat.row(j).t();
        vec count_1_j = count_1_mat.row(j).t();

        for(int l=0; l<n_cut_points; l++){

          double left_sum_inv_0 = sum(sum_inv_0_j.subvec(0, l));
          double right_sum_inv_0 = sum(sum_inv_0_j.subvec(l+1, n_cells-1));

          double left_sum_1 = sum(sum_1_j.subvec(0, l));
          double right_sum_1 = sum(sum_1_j.subvec(l+1, n_cells-1));

          double left_count_0 = sum(count_0_j.subvec(0, l));
          double right_count_0 = sum(count_0_j.subvec(l+1, n_cells-1));

          double left_count_1 = sum(count_1_j.subvec(0, l));
          double right_count_1 = sum(count_1_j.subvec(l+1, n_cells-1));


          // we need to make sure that the base measure is balanced enough
          if(left_count_0 < (double) n_min_obs_per_node || right_count_0 < (double) n_min_obs_per_node ||
               left_count_1 < (double) n_min_obs_per_node || right_count_1 < (double) n_min_obs_per_node){
            loss_left_matrix(j,l) = LARGE_NUMBER;
            loss_right_matrix(j,l) = LARGE_NUMBER;
          }else{

            // we use the two kinds of empirical measures
            double left_0_prob = left_sum_inv_0 / sum_balance_inv_0;
            double right_0_prob = right_sum_inv_0 / sum_balance_inv_0;
            double left_1_prob = left_sum_1 / sum_balance_1;
            double right_1_prob = right_sum_1 / sum_balance_1;

            // compute how much the distance increases when we cut the node using this rule
            loss_left_matrix(j,l) = pow(left_0_prob * left_1_prob, 0.5);
            loss_right_matrix(j,l) = pow(right_0_prob * right_1_prob, 0.5);

          }


        }

      }

      loss_matrix = loss_left_matrix + loss_right_matrix;

    }else{

      vec gradient_A = get_subvector_double(gradient_current, indices_current, n_total_A);

      // compute the effective size given by the weighted measures for the two groups
      mat sum_mat = zeros(d, n_cells);
      mat square_mat = zeros(d, n_cells);

      for(int j=0;j<d; j++){

        double left = left_points_A(j);
        double right = right_points_A(j);

        for(int i=0;i<n_total_A;i++){

          int index_cell = find_cell_x_is_in(residuals_A(j, i), left, right);

          sum_mat(j, index_cell) = sum_mat(j, index_cell) + gradient_A(i);
          square_mat(j, index_cell) = square_mat(j, index_cell) + pow(gradient_A(i), 2.0);

          if(group_labels_A(i) == 0){
            count_0_mat(j,index_cell) = count_0_mat(j,index_cell) + 1.0;
          }else{
            count_1_mat(j,index_cell) = count_1_mat(j,index_cell) + 1.0;
          }

        }

      }

      // calculate the loss
      for(int j=0;j<d; j++){

        vec sum_mat_j = sum_mat.row(j).t();
        vec square_mat_j = square_mat.row(j).t();

        vec count_0_j = count_0_mat.row(j).t();
        vec count_1_j = count_1_mat.row(j).t();

        for(int l=0; l<n_cut_points; l++){

          double left_sum = sum(sum_mat_j.subvec(0, l));
          double right_sum = sum(sum_mat_j.subvec(l+1, n_cells-1));

          double left_square = sum(square_mat_j.subvec(0, l));
          double right_square = sum(square_mat_j.subvec(l+1, n_cells-1));

          double left_count_0 = sum(count_0_j.subvec(0, l));
          double right_count_0 = sum(count_0_j.subvec(l+1, n_cells-1));

          double left_count_1 = sum(count_1_j.subvec(0, l));
          double right_count_1 = sum(count_1_j.subvec(l+1, n_cells-1));

          double left_count_total = left_count_0 + left_count_1;
          double right_count_total = right_count_0 + right_count_1;


          // we need to make sure that the base measure is balanced enough
          if(left_count_0 < (double) n_min_obs_per_node || right_count_0 < (double) n_min_obs_per_node ||
               left_count_1 < (double) n_min_obs_per_node || right_count_1 < (double) n_min_obs_per_node){
            loss_left_matrix(j,l) = LARGE_NUMBER;
            loss_right_matrix(j,l) = LARGE_NUMBER;
            loss_matrix(j,l) = LARGE_NUMBER;
          }else{

            // compute the sample variances
            double var_left = left_square / left_count_total - pow(left_sum / left_count_total, 2.0);
            double var_right = right_square / right_count_total - pow(right_sum / right_count_total, 2.0);

            loss_left_matrix(j,l) = var_left;
            loss_right_matrix(j,l) = var_right;

            loss_matrix(j,l) = left_count_total / (double) size_subsample * var_left + right_count_total / (double) size_subsample * var_right;

          }


        }

      }

    }

    //Decide whether to divide the current node or node
    vec loss_vec = vectorise(loss_matrix);
    uword rule_chosen = arma::index_min(loss_vec);

    // we split the current node only when the maximum improvement is positive

    if(loss_vec(rule_chosen) < node->loss){
      //now we've decided to split this node
      is_split = true;

      vec probs_rule;
      uword dim_chosen;
      int index_location_chosen;
      double location_chosen;

      double left_point;
      double right_point;
      double partition_point;

      dim_chosen = rule_chosen % d;

      index_location_chosen = rule_chosen / d;
      location_chosen = L_candidates(index_location_chosen);

      //input the information on how the current node is divided
      left_point = node->left_points(dim_chosen);
      right_point = node->right_points(dim_chosen);
      partition_point = left_point + (right_point - left_point) * location_chosen;

      node->dim_selected = dim_chosen;
      node->location = location_chosen;
      node->partition_point = partition_point;

      // make the children nodes
      // calculate "theta" based on the distribution of the pooled residuals
      // we also calculate "beta" by taking the ratio of the sum of the weights

      node->left  = get_new_node(node, true, node->dim_selected, node->location);
      node->right = get_new_node(node, false, node->dim_selected, node->location);

      for(int i=0; i<n_total_A; i++){
        int index_i = indices_current[i];

        if(residuals_A(dim_chosen, i) < partition_point){
          node->left->indices.push_back(index_i);

        }else{
          node->right->indices.push_back(index_i);
        }
      }

      // also we need to input the loss evaluated for each node
      node->left->loss = loss_left_matrix(dim_chosen, index_location_chosen);
      node->right->loss = loss_right_matrix(dim_chosen, index_location_chosen);

    }
  }


  return is_split;

}

void class_balancePM::compute_beta(Node* node){
  //input the information of the indices
  vector<int> indices_current = node->indices;
  int n_total_A = indices_current.size();

  //get more information
  ivec group_labels_A = get_subvector_int(group_labels, indices_current, n_total_A);
  vec balance_A = get_subvector_double(balance_current, indices_current, n_total_A);
  vec balance_inv_A = get_subvector_double(balance_inv_current, indices_current, n_total_A);

  //compute the effective size (sum of weights) for the current leaf node
  double sum_inv_A_0 = 0.0;
  double sum_A_1 = 0.0;

  for(int i=0; i<n_total_A; i++){
    if(group_labels_A(i) == 0){
      sum_inv_A_0 += balance_inv_A(i);
    }else{
      sum_A_1 += balance_A(i);
    }
  }

  double beta;

  // one of them might be zero, just in case
  if(sum_inv_A_0 == 0 || sum_A_1 == 0){
    beta = 1.0;
  }else{
    double beta_opt = (sum_inv_A_0 / (double) n0_learn) / (sum_A_1  / (double) n1_learn);

    beta = pow(beta_opt, learn_rate);
  }

  node->beta = beta;

}


void class_balancePM::update_balancing_weights(Node* root){

  for(int i=0; i<n;i++){
    vec temp_i = residuals_current.col(i);

    double beta_i = evaluate_density(root, temp_i);
    //NOTICE: don't forget to take the squared root
    //since we are dealing with squared root of weight
    double beta_sqrt = pow(beta_i, 0.5);

    balance_current(i) = balance_current(i) * beta_sqrt;
    balance_inv_current(i) = balance_inv_current(i) / beta_sqrt;
  }
}

double class_balancePM::evaluate_density(Node* root, vec& x){

  // note that in the current code "density" means a balancing weight

  //find a terminal node that x belongs to
  Node* curr = find_terminal_node(root, x);

  double dens_curr = curr->beta;

  return dens_curr;

}

void class_balancePM::input_indices_for_leaves(Node* root){
  for(int i=0; i<n;i++){
    vec temp_i = residuals_current.col(i);

    //find a terminal node that x belongs to
    Node* curr = find_terminal_node(root, temp_i);

    curr->indices.push_back(i);
  }
}

Node* class_balancePM::find_terminal_node(Node* root, vec& x){

  Node* curr = root;

  while(curr->left != nullptr){
    int dim_selected = curr->dim_selected;

    if(x(dim_selected) <= curr->partition_point){
      curr = curr->left;
    }else{
      curr = curr->right;
    }

  }

  return curr;

}

void class_balancePM::record_tree(Node* root){

  //Make a stack for nodes
  std::stack<Node*> stack_tree;

  //current node
  Node* curr = root;

  while(curr != nullptr || stack_tree.empty() == false){

    while(curr != nullptr){
      stack_tree.push(curr);

      if(curr->left == nullptr){
        d_store.push_back(-1);
        l_store.push_back(-1.0);
        beta_store.push_back(curr->beta);
      }else{
        //input the information
        d_store.push_back(curr->dim_selected);
        l_store.push_back(curr->location);
        beta_store.push_back(-1.0);
      }

      curr = curr->left;
    }

    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();

    curr = curr->right;
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//back-fitting functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

void class_balancePM::backfitting(){

  if(!quiet){
    Rcout << "backfitting in progress..." << "\n";
  }

  // need to consider the ratio of the sample size
  // zeta = (double) n_vec(0) / (double) n;
  n_min = std::min(n_vec(0), n_vec(1));

  // zeta0 = 0.5 * (double) n / (double) n_vec(0);
  // zeta1 = 0.5 * (double) n / (double) n_vec(1);
  zeta0 = (double) n_min / (double) n_vec(0);
  zeta1 = (double) n_min / (double) n_vec(1);

  // prior for tree updating moves
  prob_GROW = prob_moves(0);
  prob_PRUNE = prob_moves(1);
  prob_CHANGE = prob_moves(2);

  // matrix to store the estimated balancing weights (and other results)
  balance_store = zeros(n,size_backfitting);
  omega_store = zeros(size_backfitting);
  lambda_store = zeros(size_backfitting);

  // initial values are based on the output of the boosting
  balance_current = balance_boosting;
  balance_inv_current = 1.0 / balance_boosting;

  // set the hyperparameters for the sqrt of beta
  mu_prior = 1.0;

  lambda_prior = num_trees_generated_double * lambda_0;
  lambda_store.fill(lambda_prior);

  // set the initial value of omega (parameter for the Gibbs posterior approach)
  omega = 0.5;

  // repeat the back-fitting updates
  for(int index_back=-size_burnin; index_back < size_backfitting; index_back++){

    if(output_BART_ensembles){
      // prepare to save the forests
      tree_list = List();
    }

    for(int index_thin=0; index_thin < thin; index_thin++){

      for(index_tree=0; index_tree < num_trees_generated; index_tree++){

        even_or_odd = index_tree % 2;

        Node* root = root_nodes[index_tree];



        // step 0: adjust the current estimate of the balancing weights
        //         by subtracting the beta values stored in the current tree
        for(int i=0; i<n; i++){
          vec temp_i = residuals_current.col(i);

          double beta_i = evaluate_density(root, temp_i);
          double beta_sqrt = pow(beta_i, 0.5);

          balance_current(i) = balance_current(i) / beta_sqrt;
          balance_inv_current(i) = balance_inv_current(i) * beta_sqrt;
        }

        // collect leaf nodes
        std::vector<Node*> leaf_nodes;
        root->collect_leaf_nodes(leaf_nodes);

        //collect second generation internal nodes
        std::vector<Node*> SecGI_nodes;
        root->collect_SecGI_nodes(SecGI_nodes);

        n_leaf_nodes = leaf_nodes.size();
        n_SecGI_nodes = SecGI_nodes.size();

        // step 1: update the tree
        // should be prior to update beta

        // randomly select one move
        double u_move = R::runif(0.0, 1.0);
        int move;

        if(u_move < prob_GROW){
          move = 0;
        }else if(u_move < prob_GROW+prob_PRUNE){
          move = 1;
        }else{
          move = 2;
        }

        if(move == 0){
          // option 1: GROW
          // randomly choose one leaf node
          int ID_leaf_grow = random_number(n_leaf_nodes);
          GROW(leaf_nodes[ID_leaf_grow]);
        }else if(move == 1){
          // option 2: PRUNE
          // randomly choose one second generation internal node
          if(n_SecGI_nodes > 0){
            int ID_leaf_prune = random_number(n_SecGI_nodes);
            PRUNE(SecGI_nodes[ID_leaf_prune]);
          }
        }else{
          if(n_SecGI_nodes > 0){
            int ID_leaf_prune = random_number(n_SecGI_nodes);
            CHANGE(SecGI_nodes[ID_leaf_prune]);
          }
        }

        // step 2: update values of beta
        // need to recollect the leaf nodes because the structure might be different
        leaf_nodes.clear();
        root->collect_leaf_nodes(leaf_nodes);

        for(size_t i=0; i<leaf_nodes.size(); i++){
          update_beta(leaf_nodes[i]);
        }

        if(output_BART_ensembles && index_back > -1 && index_thin == thin - 1){

          //summarize the information of the current tree in the list
          d_store.clear();
          l_store.clear();

          theta_store.clear();
          beta_store.clear();

          record_tree(root);

          List list_curr_tree = Rcpp::List::create(Rcpp::Named("d") = d_store,
                                                   Rcpp::Named("l") = l_store,
                                                   Rcpp::Named("beta") = beta_store
          );

          tree_list.push_back(list_curr_tree);
        }


      }

      // update omega
      update_omega();

      // update lambda
      if(update_lambda){
        double a_post_lambda = a_prior_lambda;
        double b_post_lambda = b_prior_lambda;

        for(index_tree=0; index_tree < num_trees_generated; index_tree++){

          even_or_odd = index_tree % 2;

          Node* root = root_nodes[index_tree];

          // collect leaf nodes
          std::vector<Node*> leaf_nodes;
          root->collect_leaf_nodes(leaf_nodes);

          a_post_lambda += (double) leaf_nodes.size();

          for(size_t i=0; i<leaf_nodes.size(); i++){
            b_post_lambda += compute_post_b_lambda(leaf_nodes[i]);
          }
        }

        lambda_prior = R::rgamma(a_post_lambda/2.0, 2.0/b_post_lambda);
      }

      //store the result
      if(index_back > -1 && index_thin == thin - 1){
        balance_store.col(index_back) = balance_current;
        omega_store(index_back) = omega;
        lambda_store(index_back) = lambda_prior;

        forest_list.push_back(tree_list);

        // print the progress
        if(!quiet){
          print_progress_backfitting(index_back);
        }
      }

    }

  }

  // clear all trees
  for(Node* root : root_nodes){
   clear_node(root);
  }
}


void class_balancePM::update_beta(Node* node){

    // get the information of the current node
    vector<int> indices_A = node->indices;
    int n_total_A = indices_A.size();

    if(n_total_A == 0){
      // if this node is empty, do nothing
    }else{

      vec balance_A = get_subvector_double(balance_current, indices_A, n_total_A);
      vec balance_inv_A = get_subvector_double(balance_inv_current, indices_A, n_total_A);

      // compute the sum of balancing weight and their inverse
      ivec group_labels_A = get_subvector_int(group_labels, indices_A, n_total_A);

      double balance_inv_0_sum_A = 0.0;
      double balance_1_sum_A = 0.0;

      for(int i=0; i<n_total_A;i++){
        if(group_labels_A(i) == 0){
          balance_inv_0_sum_A += balance_inv_A(i);
        }else{
          balance_1_sum_A += balance_A(i);
        }
      }

      // generate a new value of beta
      double lambda_post;
      double mu_post;

      double beta_sq_new;

      if(even_or_odd == 0){
        // double sqrt_n = lambda_prior + 2.0 / zeta * omega * balance_inv_0_sum_A;
        // double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 / (1.0 - zeta) * omega * balance_1_sum_A;
        double sqrt_n = lambda_prior + 2.0 * zeta0 * omega * balance_inv_0_sum_A;
        double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 * zeta1* omega * balance_1_sum_A;

        mu_post = pow(sqrt_n / sqrt_d, 0.5);

        // lambda_post = lambda_prior + 2.0 / zeta * omega * balance_inv_0_sum_A;
        lambda_post = lambda_prior + 2.0 * zeta0 * omega * balance_inv_0_sum_A;

        beta_sq_new = rinversegauss_single(mu_post, lambda_post);
      }else{
        // double sqrt_n = lambda_prior + 2.0 / (1.0 - zeta) * omega * balance_1_sum_A;
        // double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 / zeta * omega * balance_inv_0_sum_A;
        double sqrt_n = lambda_prior + 2.0 * zeta1* omega * balance_1_sum_A;
        double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 * zeta0 * omega * balance_inv_0_sum_A;

        mu_post = pow(sqrt_n / sqrt_d, 0.5);

        // lambda_post = lambda_prior +  2.0 / (1.0 - zeta) * omega * balance_1_sum_A;
        lambda_post = lambda_prior +  2.0 * zeta1* omega * balance_1_sum_A;

        beta_sq_new = 1.0 / rinversegauss_single(mu_post, lambda_post);
      }

      node->beta = pow(beta_sq_new, 2.0);

      // update the current values of balancing weights
      for(int i=0; i<n_total_A;i++){

        int index = indices_A[i];
        balance_current(index) = balance_current(index) * beta_sq_new;
        balance_inv_current(index) = balance_inv_current(index)  / beta_sq_new;

      }

    }
}

void class_balancePM::GROW(Node* node){

  // get the info of the selected node
  vector<int> indices_A = node->indices;
  int n_total_A = indices_A.size();

  // no observation -> no need to split
  if(n_total_A > 0){

    // before generating candidate nodes, check the status of this node
    bool root_or_has_nieces_current = root_or_has_nieces(node);

    mat residuals_A = get_submat_double(residuals_current, indices_A, n_total_A);

    // compute the log-ML
    double log_ML_A = compute_log_ML(node);

    // randomly select dimension and cutpoint
    int dim_chosen = random_number(d);
    // int index_location_chosen = random_number(n_cut_points);
    // double location_chosen = L_candidates(index_location_chosen);
    double location_chosen = R::runif(SMALL_NUMBER, 1.0 - SMALL_NUMBER);

    // temporally cut the node
    double left_point = node->left_points(dim_chosen);
    double right_point = node->right_points(dim_chosen);
    double partition_point = left_point + (right_point - left_point) * location_chosen;

    node->dim_selected = dim_chosen;
    node->location = location_chosen;
    node->partition_point = partition_point;

    node->left  = get_new_node(node, true, node->dim_selected, node->location);
    node->right = get_new_node(node, false, node->dim_selected, node->location);

    for(int i=0; i<n_total_A; i++){
      int index_i = indices_A[i];

      if(residuals_A(dim_chosen, i) < partition_point){
        node->left->indices.push_back(index_i);
      }else{
        node->right->indices.push_back(index_i);
      }
    }

    // check if the proposed move should be accepted or not
    bool accept = false;

    // compute the log-ML for new children node
    double log_ML_A_l = compute_log_ML(node->left);
    double log_ML_A_r = compute_log_ML(node->right);

    // compute the acceptance probability
    double depth_double = (double) node->depth;
    double ratio_prior = alpha_tree * pow(1.0-alpha_tree*pow(2.0+depth_double,-beta_tree),2.0) /
      (pow(1.0+depth_double, beta_tree) - alpha_tree);


    double log_ML_ratio = log_ML_A_l + log_ML_A_r - log_ML_A;
    double ratio_n_nodes;

    if(root_or_has_nieces_current){
      ratio_n_nodes = (double) n_leaf_nodes / (((double) n_SecGI_nodes) + 1.0);
    }else{
      ratio_n_nodes = (double) n_leaf_nodes / (double) n_SecGI_nodes;
    }

    double u = R::runif(0.0, 1.0);

    // accept or reject
    if(log(u) < log(ratio_prior) + log(ratio_n_nodes) + log(prob_PRUNE) - log(prob_GROW) + log_ML_ratio){
      accept = true;
    }else{
      accept = false;
    }
  //}

    if(accept){
    }else{
      // reject: delete the new nodes
      clear_node(node->left);
      clear_node(node->right);

      node->left = nullptr;
      node->right = nullptr;
    }


  }

}

void class_balancePM::PRUNE(Node* node){

  // get the info of the leaf nodes
  vector<int> indices_A_l = node->left->indices;
  vector<int> indices_A_r = node->right->indices;

  //int n_total_A_l = indices_A_l.size();
  //int n_total_A_r = indices_A_r.size();

  // combine the two index vectors
  vector<int> indices_A = indices_A_l;
  indices_A.insert(indices_A.end(), indices_A_r.begin(), indices_A_r.end());

  // compute the log-ML for the nodes
  double log_ML_A = compute_log_ML(node);
  double log_ML_A_l = compute_log_ML(node->left);
  double log_ML_A_r = compute_log_ML(node->right);

  // compute the acceptance probability
  double depth_double = (double) node->depth;
  double ratio_prior = (pow(1.0+depth_double, beta_tree) - alpha_tree) /
                        (alpha_tree * pow(1.0-alpha_tree*pow(2.0+depth_double,-beta_tree),2.0));

  double log_ML_ratio = log_ML_A - log_ML_A_l - log_ML_A_r;
  double ratio_n_nodes = ((double) n_SecGI_nodes) / (((double) n_leaf_nodes) - 1.0);

  // accept or reject
  double u = R::runif(0.0,1.0);

  if(log(u) < log(ratio_prior) + log(ratio_n_nodes) + log(prob_GROW) - log(prob_PRUNE) + log_ML_ratio){

    // accept: delete the children nodes
    clear_node(node->left);
    clear_node(node->right);

    node->left = nullptr;
    node->right = nullptr;

    node->indices = indices_A;
  }else{
    // reject: keep the children nodes
  }

}

void class_balancePM::CHANGE(Node* node){

  // get the info of the leaf nodes
  vector<int> indices_A_l = node->left->indices;
  vector<int> indices_A_r = node->right->indices;

  int n_total_A_l = indices_A_l.size();
  int n_total_A_r = indices_A_r.size();

  // combine the two index vectors
  vector<int> indices_A = indices_A_l;
  indices_A.insert(indices_A.end(), indices_A_r.begin(), indices_A_r.end());

  int n_total_A = n_total_A_l + n_total_A_r;

  // no observation -> no need to change
  if(n_total_A > 0){

    mat residuals_A = get_submat_double(residuals_current, indices_A, n_total_A);

    // compute the log-ML for the current children node
    double log_ML_A_l_prev = compute_log_ML(node->left);
    double log_ML_A_r_prev = compute_log_ML(node->right);

    // store the previous info in case of reject
    int dim_prev = node->dim_selected;
    double location_prev = node->location;

    //vec left_points_prev = node->left_points;
    //vec right_points_prev = node->right_points;
    double partition_point_prev = node->partition_point;

    // create the candidates
    // randomly select dimension and cutpoint
    int dim_chosen = random_number(d);
    // int index_location_chosen = random_number(n_cut_points);
    // double location_chosen = L_candidates(index_location_chosen);
    double location_chosen = R::runif(SMALL_NUMBER, 1.0 - SMALL_NUMBER);

    double left_point = node->left_points(dim_chosen);
    double right_point = node->right_points(dim_chosen);
    double partition_point = left_point + (right_point - left_point) * location_chosen;

    // create the new node
    Node* left_new = get_new_node(node, true, dim_chosen, location_chosen);
    Node* right_new = get_new_node(node, false, dim_chosen, location_chosen);

    /*
    double count_0_l = 0;
    double count_0_r = 0;
    double count_1_l = 0;
    double count_1_r = 0;
    */

    for(int i=0; i<n_total_A; i++){
      int index_i = indices_A[i];

      if(residuals_A(dim_chosen, i) < partition_point){
        left_new->indices.push_back(index_i);
      }else{
        right_new->indices.push_back(index_i);
      }
    }

    // check if the proposed move should be accepted or not
    bool accept = false;

    // compute the log-ML for new children node
    double log_ML_A_l_new = compute_log_ML(left_new);
    double log_ML_A_r_new = compute_log_ML(right_new);

    // compute the acceptance probability
    double log_ML_prev = log_ML_A_l_prev + log_ML_A_r_prev;
    double log_ML_new = log_ML_A_l_new + log_ML_A_r_new;

    double log_accept_prob = log_ML_new - log_ML_prev;

    // accept or reject
    double u = R::runif(0.0,1.0);

    if(log(u) < log_accept_prob){
      accept = true;
    }else{
      accept = false;
    }

    if(accept){
      // accept: delete the previous node
      clear_node(node->left);
      clear_node(node->right);

      node->left = left_new;
      node->right = right_new;
    }else{
      // reject: delete the new nodes
      clear_node(left_new);
      clear_node(right_new);

      //recover the info of the current node
      node->dim_selected = dim_prev;
      node->location = location_prev;
      node->partition_point = partition_point_prev;
    }


  }
}

double class_balancePM::compute_log_ML(Node* node){

  // get the information of the current node
  vector<int> indices_A = node->indices;
  int n_total_A = indices_A.size();

  double out;

  if(n_total_A == 0){
    out = 0.0;
  }else{

    vec balance_A = get_subvector_double(balance_current, indices_A, n_total_A);
    vec balance_inv_A = get_subvector_double(balance_inv_current, indices_A, n_total_A);

    // compute the sum of balancing weight and their inverse
    ivec group_labels_A = get_subvector_int(group_labels, indices_A, n_total_A);

    double balance_inv_0_sum_A = 0.0;
    double balance_1_sum_A = 0.0;

    for(int i=0; i<n_total_A;i++){
      if(group_labels_A(i) == 0){
        balance_inv_0_sum_A += balance_inv_A(i);
      }else{
        balance_1_sum_A += balance_A(i);
      }
    }

    // compute the parameter values of the posterior (needed to compute the ML)
    double lambda_post;
    double mu_post;

    if(even_or_odd == 0){
      // double sqrt_n = lambda_prior + 2.0 / zeta * omega * balance_inv_0_sum_A;
      // double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 / (1.0 - zeta) * omega * balance_1_sum_A;
      double sqrt_n = lambda_prior + 2.0 * zeta0 * omega * balance_inv_0_sum_A;
      double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 * zeta1* omega * balance_1_sum_A;

      mu_post = pow(sqrt_n / sqrt_d, 0.5);

      // lambda_post = lambda_prior + 2.0 / zeta * omega * balance_inv_0_sum_A;
      lambda_post = lambda_prior + 2.0 * zeta0 * omega * balance_inv_0_sum_A;
    }else{
      // double sqrt_n = lambda_prior + 2.0 / (1.0 - zeta) * omega * balance_1_sum_A;
      // double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 / zeta * omega * balance_inv_0_sum_A;
      double sqrt_n = lambda_prior + 2.0 * zeta1* omega * balance_1_sum_A;
      double sqrt_d = lambda_prior/pow(mu_prior,2.0) + 2.0 * zeta0 * omega * balance_inv_0_sum_A;

      mu_post = pow(sqrt_n / sqrt_d, 0.5);

      // lambda_post = lambda_prior +  2.0 / (1.0 - zeta) * omega * balance_1_sum_A;
      lambda_post = lambda_prior +  2.0 * zeta1* omega * balance_1_sum_A;
    }

    out = 0.5 * (log(lambda_prior) - log(lambda_post)) +  lambda_prior / mu_prior - lambda_post / mu_post;
  }

  return out;
}

double class_balancePM::compute_post_b_lambda(Node* node){

  double beta_sqrt_current;
  if(even_or_odd == 0){
    beta_sqrt_current = pow(node->beta, 0.5);
  }else{
    beta_sqrt_current = pow(node->beta, -0.5);
  }

  double out = pow(beta_sqrt_current-mu_prior, 2.0) / (pow(mu_prior, 2.0) * beta_sqrt_current);

  return out;
}

bool class_balancePM::root_or_has_nieces(Node* node){
  if(node->parent == nullptr){
    return true;
  }else if(node->parent->left == node){
    if(node->parent->right->left != nullptr){
      return true;
    }else{
      return false;
    }
  }else{
    if(node->parent->left->left != nullptr){
      return true;
    }else{
      return false;
    }
  }
}

void class_balancePM::update_omega(){

  // obtain the posterior
  double a_post = a_prior_omega + (double) n_min;
  double b_post = b_prior_omega;

  for(int i=0; i<n;i++){
    if(group_labels(i) == 0){
      // b_post += balance_inv_current(i) / zeta;
      b_post += balance_inv_current(i) * zeta0;
    }else{
      // b_post += balance_current(i) / (1.0 - zeta);
      b_post += balance_current(i) * zeta1;
    }
  }

  // generate from the posterior
  omega = R::rgamma(a_post, 1.0 / b_post);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//miscellaneous functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////



void class_balancePM::clear_indices(Node* node){
  node->indices.clear();
}


int class_balancePM::find_cell_x_is_in(double x, double a, double b){
  double z = (x - a) / (b - a);
  double y = pow(z, alpha_cutpoint) / (pow(z, alpha_cutpoint) + pow(1.0-z, alpha_cutpoint));
  int out = y * (double) n_cells;

  // sometimes out can be n_cells if it is too close to 1
  // so need to adjust if it happens
  if(out == n_cells){
    out = n_cells - 1;
  }

  return out;
}

double class_balancePM::compute_volume(Node* node){
  vec left_points = node->left_points;
  vec right_points = node->right_points;

  return prod(right_points - left_points);
}

double class_balancePM::compute_log_volume(Node* node){
  vec left_points = node->left_points;
  vec right_points = node->right_points;

  return sum(log(right_points - left_points));
}

bool class_balancePM::does_it_have_grandchildren(Node* node){
  if(node->left == nullptr){
    return false;
  }else{
    return node->left->left != nullptr || node->right->left;
  }
}



void class_balancePM::print_progress_backfitting(int index_tree){

  double length_var = 50.0;

  if(index_tree == 0){
    Rcout << "0%" << "|--------------------------------------------------|" << "100%" << "\n";
    Rcout << "   " ;
  }

  int start = ceil(length_var * (double) index_tree / (double) size_backfitting);
  int end = ceil(length_var * (double) (index_tree + 1) / (double) size_backfitting);

  for(int i=start; i<end; i++){
    Rcout << "*" ;
  }

  if(index_tree == size_backfitting-1){
    Rcout << "  " << "\n";
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//make outputs
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

List class_balancePM::output(){


  List out;

  if(size_backfitting == 0){

    out = Rcpp::List::create(Rcpp::Named("balance_weight_boosting_data") = balance_boosting,
                             Rcpp::Named("n_nodes") = n_nodes,
                             Rcpp::Named("loss_curve") = loss_curve,
                             Rcpp::Named("tree_list") = tree_list,
                             Rcpp::Named("c") = prod_c
    );

  }else{

    out = Rcpp::List::create(Rcpp::Named("balance_weight_boosting_data") = balance_boosting,
                             Rcpp::Named("n_nodes") = n_nodes,
                             Rcpp::Named("loss_curve") = loss_curve,
                             Rcpp::Named("tree_list") = tree_list,
                             Rcpp::Named("c") = prod_c,
                             Rcpp::Named("balance_weight_BART_data") = balance_store,
                             Rcpp::Named("omega_store") = omega_store,
                             Rcpp::Named("lambda_store") = lambda_store,
                             Rcpp::Named("forest_list") = forest_list
    );

  }

  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//destructor
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

class_balancePM::~class_balancePM(){
}

void class_balancePM::clear_node(Node* root){
  if(root){
    clear_node(root->left);
    clear_node(root->right);
    delete root;
  }
}
