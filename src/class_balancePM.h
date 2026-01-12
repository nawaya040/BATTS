#ifndef ADABOOST_H
#define ADABOOST_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "helpers.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

struct Node
{
  unsigned int node_id; //1(=root),2,3,...

  int depth; //note: the depth starts from 0 (root node)

  vec left_points;
  vec right_points;

  uword dim_selected;

  double location; // the value of "L"
  double partition_point; //The partition point in the selected dimension
  //double theta; //current value of theta

  double beta; // beta (necessary only for the terminal nodes)

  vector<int> indices;

  //pointers of the parent and children nodes
  Node* parent;
  Node* left;
  Node* right;

  double loss; // quantity that needs to be minimized (definition depends on method)

  // member function to collect the lead nodes
  void collect_leaf_nodes(std::vector<Node*>& leaf_nodes){
    if (left == nullptr) {
      leaf_nodes.push_back(this);
    }else{
      left->collect_leaf_nodes(leaf_nodes);
      right->collect_leaf_nodes(leaf_nodes);
    }
  }

  void collect_SecGI_nodes(std::vector<Node*>& SecGI_nodes){
    if (left != nullptr) {
      if(left->left == nullptr && right->left == nullptr){
        SecGI_nodes.push_back(this);
      }else{
        left->collect_SecGI_nodes(SecGI_nodes);
        right->collect_SecGI_nodes(SecGI_nodes);
      }
    }
  }

};

class class_balancePM{

public:

  //Constructor
  class_balancePM(
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
  );

  //Input information
  mat X;
  ivec group_labels;
  int num_trees;
  int max_resol;
  double learn_rate;
  double split_prob;
  vec L_candidates;
  double alpha_cutpoint;
  ivec labels_train;
  double n_min_obs_per_node;
  double n_ratio_per_node;

  bool use_gradient;

  int size_burnin;
  int size_backfitting;
  int thin;

  vec prob_moves;

  double lambda_0;

  double a_prior_lambda;
  double b_prior_lambda;

  double a_prior_omega;
  double b_prior_omega;

  bool update_lambda;

  double alpha_tree;
  double beta_tree;

  bool output_BART_ensembles;
  bool quiet;

  int d;
  int n;

  mat R_current;

  int num_trees_generated;
  double num_trees_generated_double;

  int num_grid_points_L;

  ivec n_vec;
  int n_min;

  ivec indices_group_0;
  ivec indices_group_1;

  int index_tree;

  mat residuals_current;
  mat residuals_other_current;

  vec balance_current;
  vec balance_inv_current;

  mat balance_store_boosting;

  vec balance_boosting;

  vec gradient_current;

  double sum_balance_inv_0;
  double sum_balance_1;

  int n0_learn;
  int n1_learn;

  int n0_validation;
  int n1_validation;

  double prod_c;

  vector<Node*> root_nodes;

  bool is_first_stage;
  int dim_current_first;

  vector<int> indices_0;
  vector<int> indices_1;
  vec group_labels_double;

  int current_group_to_update;

  int n_cells;

  List tree_list;
  List forest_list;

  vector<int> n_nodes;

  vec loss_curve;

  vec recent_improvements;

  uvec indices_used;
  uvec indices_not_used;

  int size_subsample;

  double mu_prior;
  double lambda_prior;

  int even_or_odd;

  mat balance_store;
  vec omega_store;
  vec lambda_store;

  // double zeta;
  double zeta0;
  double zeta1;

  double omega;
  int n_cut_points;

  double prob_GROW;
  double prob_PRUNE;
  double prob_CHANGE;

  //variables to store the information of generated trees
  vector<bool> decision_store;
  vector<int> d_store;
  vector<double> l_store;
  vector<double> theta_store;
  vector<double> beta_store;

  vec tree_size;

  // variables used in the BART part

  int n_leaf_nodes;
  int n_SecGI_nodes;

  //Initialization
  void init();

  //tree functions
  Node* get_root_node();
  Node* get_new_node(Node* parent, bool this_is_left, int dim_selected, double location);

  void add_children(Node* node, int dim_selected, double location);

  void count_total_nodes(Node* node, int& count);

  //boosting functions
  void do_boosting();

  void construct_tree(Node* root);
  bool split_node(Node* node);
  void compute_beta(Node* node);

  Node* find_terminal_node(Node* root, vec& x);

  void update_balancing_weights(Node* root);
  double evaluate_density(Node* root, vec& x);

  void input_indices_for_leaves(Node* root);

  void record_tree(Node* root);

  //back-fitting algorithms
  void backfitting();
  void update_beta(Node* node);

  void GROW(Node* node);
  void PRUNE(Node* node);
  void CHANGE(Node* node);

  double compute_log_ML(Node* node);
  bool root_or_has_nieces(Node* node);

  void update_omega();
  double compute_post_b_lambda(Node* node);

  //miscellaneous functions
  void clear_indices(Node* node);

  int find_cell_x_is_in(double x, double a, double b);
  double compute_volume(Node* node);
  double compute_log_volume(Node* node);
  bool does_it_have_grandchildren(Node* node);
  void print_progress_backfitting(int index_tree);

  //output
  List output();

  //destructor
  ~class_balancePM();
  void clear_node(Node* root);

};

#endif
