#include <RcppArmadillo.h>
#include "abc_functions.h"
using namespace Rcpp;
// [[Rcpp:depends(RcppArmadillo)]]

//' @title Runs MCMC on an Ising matrix
//' @param x Ising matrix
//' @param theta Ising parameter
//' @param nits Number of iterations
// [[Rcpp::export]]
IntegerMatrix ising_chain(IntegerMatrix x, double theta, int nits) {
  int n = x.nrow();
  for(int k = 0; k < nits; k++) {
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        int n0 = x(i, j);
        int n1 = x((n + i + 1) % n, j);
        int n2 = x((n + i - 1) % n, j);
        int n3 = x(i, (n + j + 1) % n);
        int n4 = x(i, (n + j - 1) % n);
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

//' @title Converts to integer matrix
//' @param x Ising matrix
// [[Rcpp::export]]
IntegerMatrix intmat(IntegerMatrix x) {
  int n = x.nrow();
  IntegerMatrix y(n, n);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      int n0 = x(i, j);
      int n1 = x((n + i + 1) % n, j);
      int n2 = x((n + i - 1) % n, j);
      int n3 = x(i, (n + j + 1) % n);
      int n4 = x(i, (n + j - 1) % n);
      y(i, j) = n0;
    }
  }  
  return(y);
}

//' @title Generates binomial random matrix of signs
//' @param n dim
// [[Rcpp::export]]
IntegerMatrix random_sign_mat(int n) {
  IntegerMatrix x(n, n);
  NumericVector r = runif(n * n);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      if(r((i * 1) * n + j) > 0.5)
      {
        x(i, j) = 1;
      }
      else
      {
        x(i, j) = -1;
      }
    }
  }
  return(x);
}

//' @title Computes SS of Ising matrix
//' @param x Ising matrix
// [[Rcpp::export]]
int ising_ss(IntegerMatrix x) {
  int n = x.nrow();
  int s = 0;
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      int n0 = x(i, j);
      int n1 = x((n + i + 1) % n, j);
      int n2 = x((n + i - 1) % n, j);
      int n3 = x(i, (n + j + 1) % n);
      int n4 = x(i, (n + j - 1) % n);
      s = s + n0 * (n1 + n2 + n3 + n4);
    }
  }  
  return(s);
}
