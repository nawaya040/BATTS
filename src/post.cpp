// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include<bits/stdc++.h>
#include "post.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

int d_g;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List evaluate_balance_weight_boosting(List tree_list, mat eval_points){

  //initialization
  d_g = eval_points.n_cols;
  int n_eval = eval_points.n_rows;

  mat residuals_current = eval_points.t();

  vec balance_current = zeros(n_eval);
  balance_current.fill(1.0); //initial weight: all 1

  int num_trees = tree_list.size();

  for(int index_tree = 0; index_tree<num_trees; index_tree++){

    //reconstruct a tree
    Node* root = get_root_node();

    construct_tree(root, tree_list[index_tree]);

    //evaluate densities
    vec x_temp;

    for(int i=0; i<n_eval; ++i){
      x_temp = residuals_current.col(i);

      double beta_i = evaluate_density(root, x_temp);
      double beta_sqrt = pow(beta_i, 0.5);

      balance_current(i) = balance_current(i) * beta_sqrt;
    }

    //delete the current tree
    clear_node(root);

  }


  List out;

  out = Rcpp::List::create(Rcpp::Named("balance_current") = balance_current
                           );

  return out;
}

// [[Rcpp::export]]
List evaluate_balance_weight_BART(List forest_list, mat eval_points){
  //initialization
  d_g = eval_points.n_cols;
  int n_eval = eval_points.n_rows;
  int n_forests = forest_list.size();

  mat residuals_current = eval_points.t();

  mat balance_store = zeros(n_eval, n_forests);

  for(int index_forest = 0; index_forest<n_forests; index_forest++){

    List tree_list = forest_list[index_forest];

    vec balance_current = zeros(n_eval);
    balance_current.fill(1.0); //initial weight: all 1

    int num_trees = tree_list.size();

    for(int index_tree = 0; index_tree<num_trees; index_tree++){

      //reconstruct a tree
      Node* root = get_root_node();
      construct_tree(root, tree_list[index_tree]);

      //evaluate densities
      vec x_temp;

      for(int i=0; i<n_eval; ++i){
        x_temp = residuals_current.col(i);

        double beta_i = evaluate_density(root, x_temp);
        double beta_sqrt = pow(beta_i, 0.5);

        balance_current(i) = balance_current(i) * beta_sqrt;
      }

      //delete the current tree
      clear_node(root);

    }

    // store the result
    balance_store.col(index_forest) = balance_current;
  }


  List out;

  out = Rcpp::List::create(Rcpp::Named("balance_store") = balance_store
  );

  return out;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tree functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Node* get_root_node(){

  Node* new_node = new Node;

  new_node->depth = 0;

  new_node->node_id = 1;

  new_node->left_points = zeros<vec>(d_g);
  new_node->right_points = ones<vec>(d_g);

  new_node->dim_selected = 0; //Not registered
  new_node->location = 0; //Not registered
  new_node->partition_point = 0; //Not registered
  //new_node->theta = 0.0; //Not registered

  new_node->beta = 1.0;

  new_node->left = nullptr;
  new_node->right = nullptr;

  return new_node;
}


Node* get_new_node(Node* parent, bool this_is_left, int dim_selected, double location){
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
  //new_node->theta = 0.0; //Not registered

  new_node->beta = 1.0;

  new_node->left = nullptr;
  new_node->right = nullptr;

  return new_node;
}



void construct_tree(Node* node, List tree_current){

  ivec d_store = tree_current["d"];
  vec l_store = tree_current["l"];
  vec beta_store = tree_current["beta"];

  //Make a stack for nodes
  std::stack<Node*> stack_tree;

  //current node
  Node* curr = node;
  int index_node = 0;

  while(curr != nullptr || stack_tree.empty() == false){

    while(curr != nullptr){
      stack_tree.push(curr);

      //check whether or not to split the current node here

      bool is_split = (d_store(index_node) > -1);

      //split the current node if necessary
      if(is_split){
        curr->dim_selected = d_store(index_node);
        curr->location = l_store(index_node);

        curr->left = get_new_node(curr, true, curr->dim_selected, curr->location);
        curr->right = get_new_node(curr, false, curr->dim_selected, curr->location);
      }else{
        // terminal node, so need to input beta
        curr->beta = beta_store(index_node);

        curr->left = nullptr;
        curr->right = nullptr;
      }

      index_node++;

      curr = curr->left;
    }

    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();

    curr = curr->right;
  }

}

double evaluate_density(Node* root, vec& x){

  //finde a terminal node that x belongs to
  Node* curr = find_terminal_node(root, x);

  double dens_curr = curr->beta;

  return dens_curr;

}


Node* find_terminal_node(Node* root, vec& x){

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// for clearning
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
void clear_node(Node* root){

  //Make a stack for nodes
  std::stack<Node*> stack_tree;

  //current node
  Node* curr = root;

  while(curr != nullptr || stack_tree.empty() == false){

    while(curr != nullptr){
      stack_tree.push(curr);
      curr = curr->left;
    }

    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();

    Node* curr_old = curr;
    Node* curr_new = curr->right;

    delete curr_old;

    //Going up one by one, find a right node that is "alive"
    curr = curr_new;
  }

}
 */

void clear_node(Node* root){

  if(root == nullptr){
    return;
  }

  clear_node(root->left);
  root->left = nullptr;
  clear_node(root->right);
  root->right = nullptr;

  delete root;

}
