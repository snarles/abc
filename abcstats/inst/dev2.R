## Development of Rcpp functions

library(Rcpp)
library(devtools)
library(roxygen2)
library(parallel)
aa <- "abcstats"

load_all(aa)


x <- 2 * matrix(rbinom(100, 1, 0.5), 10, 10) - 1
ising_ss(x)
image(x)
image(ising_chain(x, 0.3, 1000))

####
##  Generate Ising
####

generate_data <- function(n) {
  thetas <- matrix(0, n, 1)
  ss <- matrix(0, n, 1)
  xs <- matrix(0, n, 100)
  for (i in 1:n) {
    x <- 2 * matrix(rbinom(100, 1, 0.5), 10, 10) - 1
    theta <- runif(1)/2
    x <- ising_chain(x, theta, 1000)
    xs[i, ] <- as.numeric(x)
    thetas[i] <- theta
    ss[i] <- ising_ss(x)
  }
  return(list(x = xs, y = thetas, ss = ss))
}

n <- 1000
mcc <- 4
t1 <- proc.time()
res_tr <- Reduce(list_rbind, mclapply(1:mcc, function(i) generate_data(n/mcc), mc.cores = mcc))
x_tr <- res_tr$x
y_tr <- res_tr$y
res_te <- Reduce(list_rbind, mclapply(1:mcc, function(i) generate_data(n/mcc), mc.cores = mcc))
x_te <- res_te$x
y_te <- res_te$y
proc.time() - t1
hist(res_te$ss)

####
##  Fit DNN to Ising
#### 
dims <- c(100, 500, 200, 100, 1)
Wbs <- dnn_init_slow(dims, merge = TRUE)
epoch = 30
reg = 10.0
minibatch = 25
alpha0 = 0.03
mc.cores = 4;
t1 <- proc.time()
res <- dnn_sgd_par(Wbs, x_tr, y_tr, x_te, y_te, epoch, reg, minibatch, alpha0, mc.cores)
proc.time() - t1
layout(matrix(1:2, 2, 1))
plot(res$train_costs)
plot(res$test_costs)
layout(1)
plot(y_te, res$p_te)
plot(res_te$ss, res$p_te)


