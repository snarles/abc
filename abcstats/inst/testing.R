####
##  Testing DNN code
####

####
##  Slow version
####

dims <- c(100, 500, 200, 100, 2)
Ws <- dnn_init_slow(dims)
lapply(Ws, dim) # check dimensions
x <- randn(20, 100)
yh <- dnn_predict_slow(Ws, x)
yh

####
##  Generate a training set (using GMM)
####

n <- 1000
p_x <- dims[1]
p_y <- rev(dims)[1]
k <- 10
mus_x <- randn(k, p_x)
mus_y <- randn(k, p_y)
z <- sample(k, n, TRUE)
y <- mus_y[z] + randn(n, p_y)
x <- mus_x[z] + randn(n, p_x)
# train
x_tr <- x
y_tr <- y

# test
z <- sample(k, n, TRUE)
y <- mus_y[z] + randn(n, p_y)
x <- mus_x[z] + randn(n, p_x)
x_te <- x
y_te <- y

epoch = 30
reg = 50.0
minibatch = 100
alpha0 = 0.03

Ws <- dnn_init_slow(dims)
res <- dnn_sgd_slow(Ws, x_tr, y_tr, x_te, y_te, epoch, reg, minibatch, alpha0)
layout(matrix(1:2, 2, 1))
plot(res$train_costs)
plot(res$test_costs)




