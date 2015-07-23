#include <RcppArmadillo.h>
#include "abc_functions.h"
using namespace Rcpp;
// [[Rcpp:depends(RcppArmadillo)]]

// dnn_predict -------------
//' @title Predict the output given a DNN and input matrix x
//' @param Ws Weight matrics of DNN, in a list
//' @param bs bias vectors of DNN, in a list
//' @param x Input, n x p matrix 
//' @param Hs A list to store intermediate results 
// [[Rcpp::export]]
arma::mat dnn_predict(ListOf<NumericMatrix> Ws, 
    ListOf<NumericVector> bs,
    NumericMatrix x,
    ListOf<NumericMatrix> Hs) {
  int depth = Ws.size();
  int n = x.nrow();
  int p = x.ncol();
  arma::mat y(x.begin(), n, p, false);
  //for(int i = 0; i < depth; i++) {
  int i = 0;
  SEXP temp0 = Ws[i];
  NumericMatrix temp1(temp0);
  int n1 = temp1.nrow();
  int n2 = temp1.ncol();
  arma::mat W(temp1.begin(), n1, n2, false);
  SEXP temp2 = bs[i];
  NumericVector b(temp2);
  arma::mat y2 = y*W;
//    for(int j = 0; j < n; j++) {
//      for(int k = 0; k < n2; k++) {
//        if(i == depth)
//        {
//          y(j, k) = y(j, k) + b[k];          
//        }
//        else
//        {
//          y(j, k) = tanh(y(j, k) + b[k]);        
//        }
//      }
//    }
  //}  
  return(y);
};
