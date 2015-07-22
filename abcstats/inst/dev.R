## Development of Rcpp functions

library(Rcpp)

cppFunction('
NumericVector tanhC(NumericVector x) {
  int n = x.size();
  for(int i = 0; i < n; i++) {
    x[i] = tanh(x[i]);
  }
  return(x);
}
')

tanhC(1:3)

sourceCpp('abcstats/src/dnn.cpp')

dims <- c(100, 500, 200, 100, 2)
zattach(dnn_init_slow(dims))
x <- pracma::randn(20, 100)
Hs <- dnn_make_hs(Ws, bs, x)
y <- dnn_predict(Ws, bs, x)
dim(y)
