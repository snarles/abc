## Development of Rcpp functions

library(Rcpp)
library(devtools)
library(roxygen2)

aa <- "abcstats"

load_all(aa)

dims <- c(100, 500, 200, 100, 2)
Wbs <- dnn_init_slow(dims, merge = TRUE)
dim(extract_mat(Wbs, 1))
x <- pracma::randn(20, 100)
y <- dnn_tanh(Wbs$W0, Wbs$b0, x)
y2 <- tanh(t(t(x %*% Wbs$W0) + Wbs$b0[1, ]))
sum((y2 - y)^2)
y <- dnn_predict(Wbs, x)
dim(y)
y2 <- dnn_predict_slow(Wbs[1:4], Wbs[5:8], x)
dim(y2)
f2(y2 - y)


####
##  Generate a training set (using GMM)
####

n <- 1000
p_x <- dims[1]
p_y <- rev(dims)[1]
k <- 10
mus_x <- pracma::randn(k, p_x)
mus_y <- pracma::randn(k, p_y)
z <- sample(k, n, TRUE)
y <- mus_y[z] + 0.1 * pracma::randn(n, p_y)
x <- mus_x[z] + pracma::randn(n, p_x)
# train
x_tr <- x
y_tr <- y

# test
z <- sample(k, n, TRUE)
y <- mus_y[z] + 0.1 * pracma::randn(n, p_y)
x <- mus_x[z] + pracma::randn(n, p_x)
x_te <- x
y_te <- y


yh <- dnn_predict(Wbs, x)
gr <- dnn_grad(Wbs, x, y)
gr2 <- dnn_grad_slow(Wbs[1:4], Wbs[5:8], x, y)
sapply(gr2$dWs, dim)
f2(gr$dW0 - gr2$dWs[[1]])

sapply(gr, dim)


epoch = 30
reg = 50.0
minibatch = 25
alpha0 = 0.03
mc.cores = 4;


res <- dnn_sgd_par(Wbs, x_tr, y_tr, x_te, y_te, epoch, reg, minibatch, alpha0, mc.cores)
layout(matrix(1:2, 2, 1))
plot(res$train_costs)
plot(res$test_costs)

res2 <- dnn_sgd_slow(Wbs[1:4], Wbs[5:8], x_tr, y_tr, x_te, y_te, epoch, reg, minibatch * mc.cores, alpha0)
layout(matrix(1:2, 2, 1))
plot(res2$train_costs)
plot(res2$test_costs)

