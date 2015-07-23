#include <RcppArmadillo.h>
#include "abc_functions.h"
using namespace Rcpp;
// [[Rcpp:depends(RcppArmadillo)]]

//' @title Extract arma mat from list
//' @param ll List of NumericMatrix
//' @param i Index to extract
// [[Rcpp::export]]
arma::mat extract_mat(ListOf<NumericMatrix> ll, int i) {
  SEXP temp0 = ll[i];
  NumericMatrix temp1(temp0);
  int n = temp1.nrow();
  int p = temp1.ncol();
  arma::mat x(temp1.begin(), n, p, false);
  return(x);
}