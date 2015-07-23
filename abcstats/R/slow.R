#' Initialize DNN
#' 
#' Create weight matrices for deep neural network
#' @param dims Size of each layer
#' @param sigma0 Standard deviation of initial weights
#' @param rseed Random seed
#' @return A list containing weight matrices Ws abd  biases bs
#' @export
dnn_init_slow <- function(dims, sigma0 = 0.1, rseed = 315, merge = FALSE) {
  set.seed(rseed)
  Ws <- as.list(numeric(length(dims) - 1))
  bs <- as.list(numeric(length(dims) - 1))
  for (i in 1:length(Ws)) {
    Ws[[i]] <- sigma0 * pracma::randn(dims[i], dims[i + 1])
    bs[[i]] <- rep(0, dims[i + 1])
    if (merge) bs[[i]] <- t(rep(0, dims[i+1]))
  }
  names(Ws) <- paste0("W", 1:length(Ws) - 1)
  names(bs) <- paste0("b", 1:length(Ws) - 1)
  if (merge) return(c(Ws, bs))
  list(Ws = Ws, bs = bs)
}

#' Used for interfacing with Rcpp
#' 
#' Creates list needed to store intermediate results
#' @param Ws Weight matrices of DNN
#' @param bs Bias terms of DNN
#' @param x New points, dimension n x p
#' @return An list of Hs
#' @export
dnn_make_hs <- function(Ws, bs, x) {
  n <- dim(x)[1]
  depth <- length(Ws);
  Hs <- as.list(numeric(depth));
  for (i in 1:depth) {
    Hs[[i]] <- matrix(0, n, dim(Ws[[i]])[2])
  }
  Hs
}

#' Predict given DNN and tanh transfer function
#' @param Ws Weight matrices of DNN
#' @param bs Bias terms of DNN
#' @param x New points, dimension n x p
#' @return An n x ? matrix of predictions
#' @export
dnn_predict_slow <- function(Ws, bs, x) {
  y <- x
  for (i in 1:(length(Ws) - 1)) {
    y <- tanh(t(t(y %*% Ws[[i]]) + as.numeric(bs[[i]])))
  }
  y <- t(t(y %*% Ws[[length(Ws)]]) + as.numeric(bs[[length(Ws)]]))
  y
}

#' Compute gradient for DNN
#' @param Ws Weights of DNN
#' @param bs Bias terms of DNN
#' @param x Training points, dimension n x p_x
#' @param y Training targets, dimension n x p_y
#' @return A list containing dWs and dbs
#' @export
dnn_grad_slow <- function(Ws, bs, x, y) {
  Hs <- as.list(numeric(length(Ws)))
  for (i in 1:length(Ws)) {
    if (i == 1) {  
      Hs[[i]] <- tanh(t(t(x %*% Ws[[i]]) + as.numeric(bs[[i]])))
    } else {
      Hs[[i]] <- tanh(t(t(Hs[[i - 1]] %*% Ws[[i]]) + as.numeric(bs[[i]])))
    }
  }
  delta <- t(t(Hs[[length(Hs)]] - y))
  dWs <- as.list(numeric(length(Ws)))
  dbs <- as.list(numeric(length(Ws)))
  for (i in length(Ws):2) {
    dW <- t(Hs[[i - 1]]) %*% delta
    db <- colSums(delta)
    dWs[[i]] <- dW
    dbs[[i]] <- db
    delta <- (delta %*% t(Ws[[i]])) * (1 - Hs[[i - 1]]^2)
  }
  dW <- t(x) %*% delta
  db <- colSums(delta)
  dWs[[1]] <- dW
  dbs[[1]] <- db
  list(dWs = dWs, dbs = dbs, Hs = Hs)
}

#' Computes the loss
#' @param Ws Weights of DNN
#' @param bs Bias terms of DNN
#' @param x Training points, dimension n x p_x
#' @param y Training targets, dimension n x p_y
#' @return Objective function, sqrt(sum(y - hat(y))^2/n)
#' @export
dnn_loss_slow <- function(Ws, bs, x, y) {
  yh <- dnn_predict_slow(Ws, bs, x)
  sqrt(sum((y - yh)^2)/dim(x)[1])
}

#' Optimizes the network using sgd
#' @param Ws Weights of DNN
#' @param bs Bias terms of DNN
#' @param x_tr Training points, dimension n x p_x
#' @param y_tr Training targets, dimension n x p_y
#' @param x_te Test points, dimension n x p_x
#' @param y_te Test targets, dimension n x p_y
#' @param epoch Number of epochs
#' @param reg Weight regularization
#' @param minibatch Minibatch size
#' @param alpha0 Initial step size
#' @return Objective function, sqrt(sum(y - hat(y))^2/n)
#' @export
dnn_sgd_slow <- function(Ws, bs, x_tr, y_tr, x_te, y_te, epoch = 100,
                         reg = 0.0, minibatch = 100, alpha0 = 0.1) {
  num_batch_per_epoch <- floor(dim(x_tr)[1]/minibatch)
  train_costs <- numeric(epoch)
  test_costs <- numeric(epoch)
  alphas <- alpha0 * 1/(1:epoch)
  for (i in 1:epoch) {
    for (k in 1:num_batch_per_epoch) {
      x <- x_tr[(k - 1) * minibatch + (1:minibatch), , drop = FALSE]
      y <- y_tr[(k - 1) * minibatch + (1:minibatch), , drop = FALSE]
      gr <- dnn_grad_slow(Ws, bs, x, y)
      for (ii in 1:length(Ws)) {
        dW <- -alphas[i] * (gr$dWs[[ii]] + reg * Ws[[ii]])/minibatch
        db <- -alphas[i] * (gr$dbs[[ii]])/minibatch
        Ws[[ii]] <- Ws[[ii]] + dW
        bs[[ii]] <- bs[[ii]] + db
      }
    }
    (train_costs[i] <- dnn_loss_slow(Ws, bs, x_tr, y_tr))
    (test_costs[i] <- dnn_loss_slow(Ws, bs, x_te, y_te))
  }
  list(Ws = Ws, bs = bs, train_costs = train_costs, test_costs = test_costs)
}

