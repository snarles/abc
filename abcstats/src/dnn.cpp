#include <RcppArmadillo.h>
#include "abc_functions.h"
using namespace Rcpp;
// [[Rcpp:depends(RcppArmadillo)]]

// dnn_delta_tanh -------------
//' @title Backprop subroutine for delta
//' @param W weight matrix
//' @param H H matrix
//' @param delta Input matrix 
// [[Rcpp::export]]
arma::mat dnn_delta_tanh(arma::mat W, arma::mat H, arma::mat delta) {
  arma::mat delta2 = delta*trans(W);
  int n = delta2.n_rows;
  int p = delta2.n_cols;
  for(int j = 0; j < n; j++) {
    for(int k = 0; k < p; k++) {
        delta2(j, k) = delta2(j, k) * (1 - H(j, k)*H(j, k));
    }
  }
  return(delta2);
}


// dnn_tanh -------------
//' @title Apply tanh function
//' @param W weight matrix
//' @param b bias matrix
//' @param x Input, n x p matrix 
// [[Rcpp::export]]
arma::mat dnn_tanh(arma::mat W, arma::mat b, arma::mat x) {
  int n = x.n_rows;
  int p = W.n_cols;
  arma::mat y = x*W;
  for(int j = 0; j < n; j++) {
    for(int k = 0; k < p; k++) {
        y(j, k) = tanh(y(j, k) + b(0, k));        
    }
  }
  return(y);
}

// dnn_tanh -------------
//' @title Apply linear transform
//' @param W weight matrix
//' @param b bias matrix
//' @param x Input, n x p matrix 
// [[Rcpp::export]]
arma::mat dnn_lin(arma::mat W, arma::mat b, arma::mat x) {
  int n = x.n_rows;
  int p = W.n_cols;
  arma::mat y = x*W;
  for(int j = 0; j < n; j++) {
    for(int k = 0; k < p; k++) {
        y(j, k) = y(j, k) + b(0, k);        
    }
  }
  return(y);
}



// dnn_predict -------------
//' @title Predict the output given a DNN and input matrix x
//' @param Wbs Weight matrics of DNN, in a list
//' @param x Input, n x p matrix 
// [[Rcpp::export]]
arma::mat dnn_predict(ListOf<NumericMatrix> Wbs, 
    NumericMatrix x) { 
  arma::mat W0 = extract_mat(Wbs, 0);
  arma::mat W1 = extract_mat(Wbs, 1);
  arma::mat W2 = extract_mat(Wbs, 2);
  arma::mat W3 = extract_mat(Wbs, 3);
  arma::mat b0 = extract_mat(Wbs, 4);
  arma::mat b1 = extract_mat(Wbs, 5);
  arma::mat b2 = extract_mat(Wbs, 6);
  arma::mat b3 = extract_mat(Wbs, 7);
  int n = x.nrow();
  int p = x.ncol();
  arma::mat H0(x.begin(), n, p, false);
  arma::mat H1 = dnn_tanh(W0, b0, H0);
  arma::mat H2 = dnn_tanh(W1, b1, H1);
  arma::mat H3 = dnn_tanh(W2, b2, H2);
  arma::mat H4 = dnn_lin(W3, b3, H3);
  return(H4);
};


// dnn_grad -------------
//' @title Comput the gradient given a DNN and input matrix x, target y
//' @param Wbs Weight matrics of DNN, in a list
//' @param x Input, n x p matrix 
// [[Rcpp::export]]
List dnn_grad(ListOf<NumericMatrix> Wbs, 
    NumericMatrix x, NumericMatrix y) { 
  arma::mat W0 = extract_mat(Wbs, 0);
  arma::mat W1 = extract_mat(Wbs, 1);
  arma::mat W2 = extract_mat(Wbs, 2);
  arma::mat W3 = extract_mat(Wbs, 3);
  arma::mat b0 = extract_mat(Wbs, 4);
  arma::mat b1 = extract_mat(Wbs, 5);
  arma::mat b2 = extract_mat(Wbs, 6);
  arma::mat b3 = extract_mat(Wbs, 7);
  int n = x.nrow();
  int p = x.ncol();
  arma::mat H0(x.begin(), n, p, false);
  int q = y.ncol();
  arma::mat yA(y.begin(), n, q, false);
  arma::mat H1 = dnn_tanh(W0, b0, H0);
  arma::mat H2 = dnn_tanh(W1, b1, H1);
  arma::mat H3 = dnn_tanh(W2, b2, H2);
  arma::mat H4 = dnn_lin(W3, b3, H3);
  arma::mat delta3 = H4 - yA;
  arma::mat delta2 = dnn_delta_tanh(W3, H3, delta3);
  arma::mat delta1 = dnn_delta_tanh(W2, H2, delta2);
  arma::mat delta0 = dnn_delta_tanh(W1, H1, delta1);
  arma::mat dW3 = arma::trans(H3)*delta3;
  arma::mat dW2 = arma::trans(H2)*delta2;
  arma::mat dW1 = arma::trans(H1)*delta1;
  arma::mat dW0 = arma::trans(H0)*delta0;
  arma::mat db3 = arma::sum(delta3,0);
  arma::mat db2 = arma::sum(delta2,0);
  arma::mat db1 = arma::sum(delta1,0);
  arma::mat db0 = arma::sum(delta0,0);
  return (Rcpp::List::create(
      Rcpp::Named("dW0") = dW0,
      Rcpp::Named("dW1") = dW1,
      Rcpp::Named("dW2") = dW2,
      Rcpp::Named("dW3") = dW3,
      Rcpp::Named("db0") = db0,
      Rcpp::Named("db1") = db1,
      Rcpp::Named("db2") = db2,
      Rcpp::Named("db3") = db3));
};
