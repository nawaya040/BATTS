#ifndef post_H
#define post_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

struct PredictionNode
{
  unsigned int node_id; //1(=root),2,3,...
  //maybe not necessary?

  int depth; //note: the depth starts from 0 (root node)

  vec left_points;
  vec right_points;

  uword dim_selected;

  double location; // the value of "L"
  double partition_point; //The partition point in the selected dimension
  //double theta; //current value of theta

  double beta; // beta (necessary only for the terminal nodes)

  //pointers of the parent and children nodes
  PredictionNode* left;
  PredictionNode* right;

  ~PredictionNode() {
    // just in case 1
    left_points.reset();
    right_points.reset();

    // just in case 2
    left = nullptr;
    right = nullptr;
  }

};

//Main functions
List evaluate_log_density(List tree_list, mat eval_points);

mat simulation(List tree_list, int size_simulation, mat support);
vec update_vec(PredictionNode* node, vec& x);
//List evaluate_utility(List tree_list, mat eval_points, mat support);

//Tree functions
PredictionNode* get_root_node();
PredictionNode* get_new_node(PredictionNode* parent, bool this_is_left, int dim_selected, double location);

void construct_tree(PredictionNode* node, List tree_current);

double evaluate_density(PredictionNode* root, vec& x);

PredictionNode* find_terminal_node(PredictionNode* root, vec& x);

//for clearning
void clear_node(PredictionNode* root);

#endif
