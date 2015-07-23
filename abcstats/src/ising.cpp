#include <RcppArmadillo.h>
#include "abc_functions.h"
using namespace Rcpp;
// [[Rcpp:depends(RcppArmadillo)]]

//' @title Runs MCMC on an Ising matrix
//' @param x Ising matrix
//' @param theta Ising parameter
//' @param nits Number of iterations
// [[Rcpp::export]]
NumericMatrix ising_chain(NumericMatrix x, double theta, int nits) {
  int n = x.nrow();
  for(int k = 0; k < nits; k++) {
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        int n0 = x(i, j);
        int n1 = x((i + 1) % n, j);
        int n2 = x((i - 1) % n, j);
        int n3 = x(i, (j + 1) % n);
        int n4 = x(i, (j - 1) % n);
        int s = n0 * (n1 + n2 + n3 + n4);
        double prob = exp(-2 * theta * s);
        double roll = runif(1)(0);
        if (roll < prob) {
          x(i, j) = -n0;
        }
      }
    }  
  }
  return(x);
}