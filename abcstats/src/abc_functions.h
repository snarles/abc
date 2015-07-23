#ifndef _ABC_FUNCTIONS_HPP
#define _ABC_FUNCTIONS_HPP

#include <RcppArmadillo.h>
using namespace Rcpp;

// declare utility functions
arma::mat extract_mat(ListOf<NumericMatrix> ll, int i);

#endif // __ABC_FUNCTIONS_HPP