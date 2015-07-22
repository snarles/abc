####
##  Testing DNN code
####

####
##  Slow version
####

dims <- c(100, 500, 200, 100, 2)
temp <- dnn_init_slow(dims)
zattach(temp)
lapply(Ws, dim) # check dimensions
x <- pracma::randn(20, 100)
yh <- dnn_predict_slow(Ws, bs, x)
yh

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

Ws <- dnn_init_slow(dims)
gr <- dnn_grad_slow(Ws, bs, x_tr, y_tr)

epoch = 30
reg = 50.0
minibatch = 100
alpha0 = 0.03


res <- dnn_sgd_slow(Ws, bs, x_tr, y_tr, x_te, y_te, epoch, reg, minibatch, alpha0)
layout(matrix(1:2, 2, 1))
plot(res$train_costs)
plot(res$test_costs)




